# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 The Aibrix Team.
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

import fcntl
import json
import os
import tempfile
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .exporters import JsonlExporter
from .schema import TraceEvent


@dataclass(frozen=True, slots=True)
class TraceRunStatus:
    valid: bool = True
    reasons: tuple[str, ...] = ()
    dropped_events: int = 0


class NullRecorder:
    """OFF never calls an event factory or allocates an event record."""

    def record(self, event_or_factory) -> bool:
        return False

    def invalidate(self, reason: str) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(slots=True)
class _Shard:
    events: deque[TraceEvent] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    accepted: int = 0
    exported: int = 0


class ShardedRingRecorder:
    """Bounded per-thread queues retain entries until a successful export.

    Record performs no JSON encoding or file IO. Brief locks protect queue
    metadata; the exporter never holds them across a write. Flush waits for
    events accepted before the call, or returns once export has failed.
    """

    def __init__(self, *, capacity: int, output: str | Path):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._local = threading.local()
        self._condition = threading.Condition()
        self._shards: list[_Shard] = []
        self._status = TraceRunStatus()
        self._accepted = 0
        self._exported = 0
        self._closed = False
        self._export_failed = False
        self._output = Path(output)
        self._owner_path = self._output.with_suffix(
            self._output.suffix + ".owner"
        )
        self._guard_path = self._output.with_suffix(
            self._output.suffix + ".owner.guard"
        )
        self._conflict_path = self._output.with_suffix(
            self._output.suffix + ".owner-conflict"
        )
        self._owns_output = False
        self._owner_token = uuid.uuid4().hex
        try:
            self._claim_output()
        except OSError:
            self._invalidate("owner_claim_error")
        self._exporter = (
            JsonlExporter(self._output) if self._owns_output else None
        )
        self._worker = threading.Thread(
            target=self._export_loop,
            name=f"compression-trace-{id(self)}",
            daemon=True,
        )
        self._worker.start()

    def _claim_output(self) -> None:
        self._guard_path.parent.mkdir(parents=True, exist_ok=True)
        guard_fd = os.open(self._guard_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX)
            owner_exists = self._owner_path.exists()
            previous = None
            if owner_exists:
                try:
                    previous = json.loads(self._owner_path.read_text())
                except (OSError, ValueError):
                    previous = None
            if owner_exists and (
                not isinstance(previous, dict)
                or previous.get("state") != "closed"
            ):
                self._invalidate("multiple_recorder_owners")
                conflict_fd = os.open(
                    self._conflict_path,
                    os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                os.close(conflict_fd)
                return
            payload = {
                "state": "active",
                "pid": os.getpid(),
                "token": self._owner_token,
            }
            with self._owner_path.open("w") as owner:
                json.dump(payload, owner)
                owner.flush()
                os.fsync(owner.fileno())
            (self._output.with_name("run-status.json")).unlink(missing_ok=True)
            self._conflict_path.unlink(missing_ok=True)
            self._owns_output = True
        finally:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
            os.close(guard_fd)

    @property
    def status(self) -> TraceRunStatus:
        with self._condition:
            return self._status

    def _invalidate(self, reason: str, *, dropped: bool = False) -> None:
        # Caller owns the condition; publish a consistent immutable snapshot.
        reasons = self._status.reasons
        if reason not in reasons:
            reasons += (reason,)
        self._status = TraceRunStatus(
            False, reasons, self._status.dropped_events + int(dropped)
        )

    def invalidate(self, reason: str) -> None:
        """Publish an observation failure without interrupting serving."""
        with self._condition:
            self._invalidate(reason)
            self._condition.notify_all()

    def _shard_for_current_thread(self) -> _Shard:
        shard = getattr(self._local, "shard", None)
        if shard is None:
            shard = _Shard()
            with self._condition:
                self._shards.append(shard)
            self._local.shard = shard
        return shard

    def record(
        self, event_or_factory: TraceEvent | Callable[[], TraceEvent]
    ) -> bool:
        if not self._owns_output:
            return False
        shard = self._shard_for_current_thread()
        with self._condition:
            if self._closed:
                self._invalidate("recorder_closed", dropped=True)
                return False
            with shard.lock:
                if len(shard.events) >= self._capacity:
                    self._invalidate("dropped_event", dropped=True)
                    return False
        try:
            event = (
                event_or_factory()
                if callable(event_or_factory)
                else event_or_factory
            )
            # Exact schema objects guarantee scalar-only ownership, including
            # no subclasses that attach a KV buffer or MemoryRegion.
            if type(event) is not TraceEvent:
                raise TypeError("recorder accepts TraceEvent metadata only")
        except Exception:
            with self._condition:
                self._invalidate("record_error", dropped=True)
            return False
        with self._condition:
            if self._closed:
                self._invalidate("recorder_closed", dropped=True)
                return False
            with shard.lock:
                # A factory may reenter record on this same thread.
                if len(shard.events) >= self._capacity:
                    self._invalidate("dropped_event", dropped=True)
                    return False
                shard.events.append(event)
            shard.accepted += 1
            self._accepted += 1
            self._condition.notify_all()
        return True

    def flush(self) -> None:
        with self._condition:
            # A later event from an earlier-snapshotted shard must not stand in
            # for a pre-flush event in another shard that is still pending.
            targets = tuple((shard, shard.accepted) for shard in self._shards)
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: self._export_failed
                or all(shard.exported >= target for shard, target in targets)
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.flush()
        self._worker.join()
        if self._owns_output:
            self._close_owner_claim()

    def _close_owner_claim(self) -> None:
        guard_fd = os.open(self._guard_path, os.O_RDWR)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX)
            payload = json.loads(self._owner_path.read_text())
            if payload.get("token") == self._owner_token:
                payload["state"] = "closed"
                with self._owner_path.open("w") as owner:
                    json.dump(payload, owner)
                    owner.flush()
                    os.fsync(owner.fileno())
        except (OSError, ValueError):
            self._invalidate("owner_close_error")
        finally:
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
            os.close(guard_fd)

    def _publish_status(self) -> None:
        status_path = self._output.with_name("run-status.json")
        with self._condition:
            payload = {
                "valid": self._status.valid,
                "reasons": list(self._status.reasons),
                "dropped_events": self._status.dropped_events,
            }
        temporary = None
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{status_path.name}.",
                suffix=".tmp",
                dir=status_path.parent,
            )
            temporary = Path(name)
            with os.fdopen(fd, "w") as output:
                output.write(json.dumps(payload, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, status_path)
            directory = os.open(status_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            with self._condition:
                self._invalidate("run_status_publish_error")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _export_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed or self._accepted > self._exported
                    )
                    if self._accepted == self._exported and self._closed:
                        return
                    shards = tuple(self._shards)
                batches = []
                for shard in shards:
                    with shard.lock:
                        batches.append((shard, tuple(shard.events)))
                events = [event for _, batch in batches for event in batch]
                assert self._exporter is not None
                self._exporter.write_batch(events)
                for shard, batch in batches:
                    with shard.lock:
                        for _ in batch:
                            shard.events.popleft()
                with self._condition:
                    for shard, batch in batches:
                        shard.exported += len(batch)
                    self._exported += len(events)
                    self._condition.notify_all()
        except Exception:
            with self._condition:
                self._invalidate("export_error")
                self._export_failed = True
                self._condition.notify_all()
        finally:
            try:
                if self._exporter is not None:
                    self._exporter.close()
            except Exception:
                with self._condition:
                    self._invalidate("export_error")
                    self._export_failed = True
                    self._condition.notify_all()
            if self._owns_output:
                if self._conflict_path.exists():
                    with self._condition:
                        self._invalidate("multiple_recorder_owners")
                self._publish_status()
