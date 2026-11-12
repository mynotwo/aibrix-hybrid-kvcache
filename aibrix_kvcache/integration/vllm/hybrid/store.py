# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker-side record store: one flat record per boundary on AIBrix L2.

Records are ``record.RecordCodec`` envelopes. A load stages the payload in
a pinned host buffer in parallel chunks, copies each chunk to the GPU as soon
as it is staged, decodes it there (ANS), checks the checksum on the GPU and
only then writes the blocks, with device-to-device copies.

``save_async`` only enqueues GPU work and records an event, so the worker
never waits for the GPU. On the current stream, ahead of any kernel that
reuses the blocks, it gathers them into one device buffer and checksums it;
then, on a side stream the worker's stream does not wait for, it copies the
buffer into a pinned staging buffer. One background thread waits for the
event, encodes and stores the record (ANS: encodes the device buffer frame
by frame and copies only that); it uses only GIL-releasing primitives on
bulk data so it does not stall the worker thread. At most ``max_inflight``
staging buffers exist; while all are busy a save is dropped, never waited
for, so a slow save costs at most its own record, not the engine. A key
already stored is not stored again: any record of a key restores the same
tokens, and backends refuse to replace one.

With an ``OperationTrace``, a save records KV_EXPORT -> CODEC_ENCODE ->
L2_PUT and a load L2_GET -> CODEC_DECODE -> KV_IMPORT, each stage depending
on the previous one, in the compression profiler's byte accounting.
"""

import asyncio
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from typing import Iterator, Mapping, NamedTuple, Optional, Sequence

import torch

from ....compression_profiler import EventType
from ....compression_profiler.operation import OperationTrace
from ....l2.connectors.connector import Connector
from .gpu_codec import AnsCodec, checksum
from .layout import HybridLayout
from .profiling import clock, device_event, device_window, stage
from .record import CHECKSUMS, FRAMING, RecordCodec, payload_tensor, pin


class Fetched(NamedTuple):
    flat: torch.Tensor  # the record on the device
    decoded: tuple[str, ...]  # its CODEC_DECODE stage
    stored_bytes: int
    transferred_bytes: int  # host-to-device
    start: int  # host clock when decoding started
    began: Optional[torch.cuda.Event]  # and on the device


class HybridStoreError(RuntimeError):
    """A record could not be saved or restored; the caller must recompute."""


class HybridStore:
    def __init__(
        self,
        layout: HybridLayout,
        backend: Connector,
        codec: str,
        *,
        load_threads: int = 16,
        chunk_bytes: int = 1 << 20,
        max_inflight: int = 4,
        pin_records: bool = False,
    ):
        """``pin_records``: the backend keeps the records it is given in this
        process (MEMORY). They are then page-locked, so a load copies them
        to the GPU by DMA as they are, with no staging copy."""
        self.layout = layout
        self.backend = backend
        self.codec = codec
        self.signature = layout.signature()
        self._record = RecordCodec(
            codec,
            self.signature,
            # Two chunks per load thread balance the threads.
            chunks=2 * load_threads,
            chunk_bytes=chunk_bytes,
        )
        self._device = layout.slots[0].tensor.device
        self._cuda = self._device.type == "cuda"
        self._ans = AnsCodec() if codec == "ANS" else None
        # Device-to-host copies and encodings, off the worker's stream; and
        # host-to-device copies of loads, which ANS decodes overlap.
        self._side = torch.cuda.Stream(self._device) if self._cuda else None
        self._upload = torch.cuda.Stream(self._device) if self._cuda else None
        self._host = torch.empty(0, dtype=torch.uint8)
        self._local = threading.local()
        self._saver = ThreadPoolExecutor(1, thread_name_prefix="hybrid-save")
        self._loader = ThreadPoolExecutor(
            load_threads, thread_name_prefix="hybrid-load"
        )
        self._pin_records = pin_records and self._cuda
        self._free: queue.Queue = queue.Queue()
        self._max_inflight = max_inflight
        self.staging_buffers = 0
        self.staging_allocs = 0

    def _run(self, coro):
        # One event loop per calling thread (worker or background saver).
        loop = getattr(self._local, "loop", None)
        if loop is None:
            loop = self._local.loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)

    def _pinned(self, size: int) -> torch.Tensor:
        return torch.empty(size, dtype=torch.uint8, pin_memory=self._cuda)

    def _host_buffer(self, size: int) -> torch.Tensor:
        """Reusable (pinned on CUDA) host buffer for loads."""
        if self._host.numel() < size:
            self._host = self._pinned(size)
        return self._host[:size]

    def _take_staging(self, size: int) -> torch.Tensor:
        """A free staging buffer; none while ``max_inflight`` are busy."""
        try:
            buf = self._free.get_nowait()
        except queue.Empty:
            if self.staging_buffers >= self._max_inflight:
                raise HybridStoreError("save dropped: earlier saves busy")
            self.staging_buffers += 1
            self.staging_allocs += 1
            buf = self._pinned(size)
        if buf.numel() < size:
            self.staging_allocs += 1
            buf = self._pinned(size)
        return buf

    def save(
        self,
        key: bytes,
        block_ids: Mapping[int, Sequence[int]],
        trace: Optional[OperationTrace] = None,
    ) -> dict:
        return self.save_async(key, block_ids, trace).result()

    def save_async(
        self,
        key: bytes,
        block_ids: Mapping[int, Sequence[int]],
        trace: Optional[OperationTrace] = None,
    ) -> Future:
        """Stage the blocks now; encode and store them in the background.

        The result is ``{"raw_bytes", "stored_bytes", "put"}``, ``put`` being
        the L2_PUT operation id (empty without a trace).
        """
        enqueued = clock(trace)
        size = sum(self.layout.gather_sizes(block_ids))
        buf = self._take_staging(size)
        timed = self._cuda and trace is not None
        began = device_event(timed, True)
        try:
            # Freed for reuse once its last reader on this stream has run.
            flat = torch.empty(size, dtype=torch.uint8, device=self._device)
            self.layout.gather_into(block_ids, flat)
            check = torch.empty(2, dtype=torch.int64, pin_memory=self._cuda)
            check.copy_(checksum(flat), non_blocking=True)
            with self._off_worker_stream(flat):
                if self._ans is None:
                    buf[:size].copy_(flat, non_blocking=True)
                done = device_event(self._cuda, timed)
        except Exception as exc:
            self._free.put(buf)
            # E.g. out of device memory: lose this save, not the engine.
            raise HybridStoreError(f"save staging failed: {exc}") from exc
        ans = flat if self._ans is not None else None
        return self._saver.submit(
            self._encode_put,
            key,
            buf,
            size,
            check,
            ans,
            began,
            done,
            trace,
            enqueued,
        )

    @contextmanager
    def _off_worker_stream(self, flat: torch.Tensor) -> Iterator[None]:
        """Run what follows on the side stream, once the worker's stream
        has produced ``flat``."""
        if not self._cuda:
            yield
            return
        self._side.wait_stream(torch.cuda.current_stream(self._device))
        flat.record_stream(self._side)
        with torch.cuda.stream(self._side):
            yield

    # The worker made the staging buffers, under vLLM's inference mode.
    @torch.inference_mode()
    def _encode_put(
        self, key, buf, size, check, flat, began, done, trace, enqueued
    ) -> dict:
        try:
            if self._run(self.backend.exists(key)).is_ok():
                return self._already_stored(key, size, trace)
            # Before this, the save waited behind earlier saves (queue);
            # after, for the step's kernels and then the copies themselves.
            start = clock(trace)
            if done is not None:
                done.synchronize()
            host = buf[:size]
            frames: tuple[list[int], int] = ([], 0)
            if flat is not None:
                buf, frames = self._ans_encode(flat, buf)
                host = buf[: sum(frames[0])]
            exported = stage(
                trace,
                EventType.KV_EXPORT,
                key,
                start,
                trace.context.dependency_ids if trace else (),
                enqueue_ns=enqueued,
                # GPU time of the gather and copy alone (an ANS encoding is
                # in the host window); before it, the step's kernels ran.
                **device_window(began, done),
                phase="d2h",
                raw_bytes=size,
                transferred_bytes=host.numel(),
                result="copied",
            )
            return self._encode_put_staged(
                key, host, size, check.tolist(), frames, trace, exported
            )
        finally:
            self._free.put(buf)
            if trace is not None:
                trace.finish()

    def _ans_encode(self, flat, buf) -> tuple[torch.Tensor, tuple]:
        """``flat``'s ANS frames end to end in ``buf`` or, for a tiny record
        they outgrow, a larger buffer; it, frame lengths and raw frame size.
        Under the store's device: nvCOMP waits there for each frame's size.
        """
        assert self._ans is not None
        lengths, at = [], 0
        with torch.cuda.device(self._device), torch.cuda.stream(self._side):
            for frame in self._ans.encode(flat):
                if at + frame.numel() > buf.numel():
                    self.staging_allocs += 1
                    self._side.synchronize()
                    grown = self._pinned(2 * (at + frame.numel()))
                    grown[:at], buf = buf[:at], grown
                buf[at : at + frame.numel()].copy_(frame, non_blocking=True)
                lengths.append(frame.numel())
                at += frame.numel()
            self._side.synchronize()
        return buf, (lengths, self._ans.frame_bytes)

    def _already_stored(self, key, size, trace) -> dict:
        after = trace.context.dependency_ids if trace else ()
        put = stage(
            trace,
            EventType.L2_PUT,
            key,
            clock(trace),
            after,
            destination_tier="L2",
            connector_write_bytes=0,
            stored_object_bytes=0,
            result="hit",
        )
        return {
            "raw_bytes": size,
            "stored_bytes": 0,
            "put": put[0] if put else "",
        }

    def _encode_put_staged(
        self, key, host, size, check, frames, trace, exported
    ) -> dict:
        """Store ``host``: the flat bytes, or ``frames`` encoding ``size``
        flat bytes whose checksum is ``check``."""
        start, cpu = clock(trace), time.process_time_ns()
        envelope, payload, meta = self._record.encode(
            key, memoryview(host.numpy()), size, check, frames
        )
        if self._pin_records:
            pin(envelope)
        encoded = stage(
            trace,
            EventType.CODEC_ENCODE,
            key,
            start,
            exported,
            raw_bytes=size,
            payload_bytes=payload,
            metadata_bytes=meta,
            framing_bytes=FRAMING,
            envelope_bytes=len(envelope),
            checksum_algorithm=CHECKSUMS[self.codec],
            cpu_process_start_ns=cpu,
            cpu_process_end_ns=time.process_time_ns(),
            result="encoded",
        )
        start = clock(trace)
        status = self._run(self.backend.put_bytes(key, envelope))
        put = stage(
            trace,
            EventType.L2_PUT,
            key,
            start,
            encoded,
            destination_tier="L2",
            envelope_bytes=len(envelope),
            connector_write_bytes=len(envelope) if status.is_ok() else None,
            stored_object_bytes=len(envelope) if status.is_ok() else None,
            result="stored" if status.is_ok() else "store_error",
        )
        if not status.is_ok():
            raise HybridStoreError(f"put failed: {status}")
        return {
            "raw_bytes": size,
            "stored_bytes": len(envelope),
            "put": put[0] if put else "",
        }

    def _fetch(
        self,
        key: bytes,
        block_ids: Mapping[int, Sequence[int]],
        trace: Optional[OperationTrace],
        after: tuple[str, ...],
        phase: str,
    ) -> Fetched:
        """Read a record, then stage, decode and check it on the device."""
        start = clock(trace)
        status = self._run(self.backend.get_bytes(key))
        envelope = memoryview(status.get()) if status.is_ok() else None
        size = len(envelope) if envelope is not None else None
        got = stage(
            trace,
            EventType.L2_GET,
            key,
            start,
            after,
            phase=phase,
            source_tier="L2",
            connector_read_bytes=size,
            retrieved_object_bytes=size,
            result="loaded" if status.is_ok() else "not_found",
        )
        if envelope is None:
            raise HybridStoreError(f"record not found: {status}")
        expected = sum(self.layout.gather_sizes(block_ids))
        start, cpu = clock(trace), time.process_time_ns()
        began = device_event(self._cuda and trace is not None, True)
        jobs: list[Future] = []
        try:
            header = self._record.parse(key, envelope, expected)
            host = payload_tensor(header)
            if not host.is_pinned():
                host = self._host_buffer(host.numel())
                jobs, step = self._record.stage(
                    header, host.numpy(), self._loader
                )
            else:
                # A pinned record goes to the device by DMA as it is.
                step = self._record.step(host.numel())
            staged, ready = self._upload_chunks(host, step, jobs)
            flat = staged
            if self._ans is not None:
                flat = self._ans.decode(staged, header.frames, expected, ready)
            elif ready[-1][1] is not None:
                torch.cuda.current_stream().wait_event(ready[-1][1])
            if checksum(flat).tolist() != header.checksum:
                raise ValueError("checksum mismatch")
        except torch.OutOfMemoryError as exc:  # recompute, keep the engine
            raise HybridStoreError(f"restore staging failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            stage(
                trace,
                EventType.CODEC_DECODE,
                key,
                start,
                got,
                phase=phase,
                result="integrity_error",
            )
            raise HybridStoreError(f"record integrity: {exc}") from exc
        finally:
            # The host buffer is reused: nothing may still read or write it.
            for job in jobs:
                job.cancel()
            wait(jobs)
            if self._cuda:
                self._upload.synchronize()
                torch.cuda.current_stream().synchronize()
        decoded = stage(
            trace,
            EventType.CODEC_DECODE,
            key,
            start,
            got,
            phase=phase,
            raw_bytes=expected,
            decode_output_bytes=expected,
            payload_bytes=len(header.payload),
            metadata_bytes=header.meta_bytes,
            framing_bytes=FRAMING,
            envelope_bytes=len(envelope),
            checksum_algorithm=CHECKSUMS[self.codec],
            cpu_process_start_ns=cpu,
            cpu_process_end_ns=time.process_time_ns(),
            result="decoded",
        )
        return Fetched(flat, decoded, len(envelope), host.numel(), start, began)

    def _upload_chunks(self, host, step, jobs):
        """``host`` copied to the device by chunks of ``step``, each once
        staged (``jobs``), on the upload stream; and per chunk its offset
        and an event after its copy (None off CUDA)."""
        staged = torch.empty_like(host, device=self._device)
        ready = []
        if self._cuda:
            # In order after the work before the load, as on one stream.
            self._upload.wait_stream(torch.cuda.current_stream())
            staged.record_stream(self._upload)
        context = (
            torch.cuda.stream(self._upload) if self._cuda else nullcontext()
        )
        with context:
            for i, at in enumerate(range(0, max(host.numel(), 1), step)):
                if jobs:
                    jobs[i].result()
                staged[at : at + step].copy_(
                    host[at : at + step], non_blocking=True
                )
                ready.append((at, device_event(self._cuda, False)))
        return staged, ready

    def load(
        self,
        key: bytes,
        block_ids: Mapping[int, Sequence[int]],
        *,
        readback: bool = False,
        trace: Optional[OperationTrace] = None,
        after: tuple[str, ...] = (),
    ) -> dict:
        """Restore a record into ``block_ids``; ``after``: what it awaited."""
        try:
            fetched = self._fetch(key, block_ids, trace, after, "restore")
            flat = fetched.flat
            start = clock(trace)
            self.layout.scatter_flat(block_ids, flat)
            ended = device_event(fetched.began is not None, True)
            if self._cuda:
                torch.cuda.synchronize()
            # The restore is done; a diagnostic readback is not part of it.
            end = clock(trace)
            stats = {
                "raw_bytes": flat.numel(),
                "stored_bytes": fetched.stored_bytes,
            }
            if readback:
                # Diagnostic: the destination blocks now hold the record.
                live = self.layout.gather_flat(block_ids)
                stats["readback_equal"] = torch.equal(live, flat)
            stage(
                trace,
                EventType.KV_IMPORT,
                key,
                start,
                fetched.decoded,
                end=end,
                # The device work began with the copies to the device.
                enqueue_ns=fetched.start,
                phase="restore",
                raw_bytes=flat.numel(),
                transferred_bytes=fetched.transferred_bytes,
                **device_window(fetched.began, ended),
                result="restored",
                restore_result={
                    True: "readback_equal",
                    False: "readback_differs",
                    None: "restored",
                }.get(stats.get("readback_equal"), "restored"),
            )
            return stats
        finally:
            if trace is not None:
                trace.finish()

    def compare(
        self,
        key: bytes,
        block_ids: Mapping[int, Sequence[int]],
        trace: Optional[OperationTrace] = None,
    ) -> list[dict]:
        """Diagnostic: per-slot equality of a record and live blocks."""
        try:
            fetched = self._fetch(key, block_ids, trace, (), "verify")
            start = clock(trace)
            stored = fetched.flat.cpu().numpy().tobytes()
            report, at = [], 0
            for slot, current in zip(
                self.layout.slots, self.layout.gather(block_ids)
            ):
                report.append(
                    {
                        "group_id": slot.group_id,
                        "name": slot.name,
                        "tensor_index": slot.tensor_index,
                        "equal": stored[at : at + len(current)] == current,
                    }
                )
                at += len(current)
            differing = sum(not r["equal"] for r in report)
            stage(
                trace,
                EventType.KV_IMPORT,
                key,
                start,
                fetched.decoded,
                phase="verify",
                raw_bytes=len(stored),
                result="compared",
                restore_result="equal"
                if not differing
                else f"differs:{differing}/{len(report)}",
            )
            return report
        finally:
            if trace is not None:
                trace.finish()

    def close(self) -> None:
        self._saver.shutdown(wait=True)
        self._loader.shutdown(wait=True)
