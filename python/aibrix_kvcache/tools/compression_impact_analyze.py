#!/usr/bin/env python3
"""Validate v1 traces and attribute observed host latency via explicit DAGs.

Parent edges describe ownership/start causality, not necessarily completion.
Dependency edges describe completion; overlapping batch/barrier windows are
clipped at that completion. No edge is inferred from timestamp proximity.
Host spans are elapsed windows, never CPU time or measured device duration.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from graphlib import TopologicalSorter
from pathlib import Path

from aibrix_kvcache.compression_profiler.recorder import TraceRunStatus
from aibrix_kvcache.compression_profiler.schema import TraceEvent
from aibrix_kvcache.compression_profiler.validation import (
    ValidationReport,
    validate_byte_accounting,
    validate_trace,
)


class TraceValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(", ".join(report.reasons))


@dataclass
class OperationDAG:
    records: dict[str, dict]
    predecessors: dict[str, set[str]]
    order: tuple[str, ...]
    validation: ValidationReport


@dataclass
class OperationTiming:
    queue_ns: int | None
    active_ns: int
    overlap_ns: int
    critical_ns: int = 0
    slack_ns: int | None = None


@dataclass
class CriticalPath:
    terminal_event: str
    total_ns: int
    queue_ns: int | None
    observed_queue_ns: int
    active_ns: int
    overlap_ns: int
    unobserved_ns: int
    theoretical_lower_bound_ns: int
    lower_bound_components_ns: dict[str, int]
    by_operation: dict[str, OperationTiming]


def build_operation_dag(records, status=None):
    records = list(records)
    structural = validate_trace(
        records, status or TraceRunStatus(), require_complete=True
    )
    if not structural.valid:
        raise TraceValidationError(structural)
    accounting = validate_byte_accounting(records)
    if not accounting.valid:
        raise TraceValidationError(accounting)
    by_id = {r["operation_id"]: dict(r) for r in records}
    graph = {
        key: set(row.get("dependency_ids", ())) for key, row in by_id.items()
    }
    for key, row in by_id.items():
        parent = row.get("parent_operation_id")
        if parent:
            graph[key].add(parent)
        for dep in row.get("dependency_ids", ()):
            if by_id[dep]["end_ns"] > row["end_ns"]:
                raise TraceValidationError(
                    ValidationReport(False, ("dependency_completion_order",))
                )
        if parent and by_id[parent]["start_ns"] > row["start_ns"]:
            raise TraceValidationError(
                ValidationReport(False, ("parent_start_order",))
            )
    return OperationDAG(
        by_id, graph, tuple(TopologicalSorter(graph).static_order()), accounting
    )


def _begin(row):
    enqueue = row.get("enqueue_ns")
    return row["start_ns"] if enqueue is None else enqueue


def _causal_states(dag, terminal):
    # An ownership prefix does not wait for its parent's later completion.
    # Full-graph validation has already checked every edge, including cycles.
    found, pending = {}, [(terminal, dag.records[terminal]["end_ns"])]
    while pending:
        state = pending.pop()
        if state in found:
            continue
        operation, limit = state
        row = dag.records[operation]
        completed = {(p, dag.records[p]["end_ns"])
                     for p in row.get("dependency_ids", ())
                     if dag.records[p]["end_ns"] <= limit}
        owner, parent = None, row.get("parent_operation_id")
        if parent:
            prior = dag.records[parent]
            prefix = min(prior["end_ns"], _begin(row), limit)
            if prior["end_ns"] <= prefix:
                completed.add((parent, prefix))
            elif _begin(prior) <= prefix:
                owner = (parent, prefix)
        found[state] = (completed, owner)
        pending.extend(completed)
        if owner:
            pending.append(owner)
    position = {op: i for i, op in enumerate(dag.order)}
    return found, sorted(found, key=lambda s: (position[s[0]], s[1]))


def _overlap_intervals(records):
    changes = {}
    for row in records:
        start, end = row["start_ns"], row["end_ns"]
        changes[start] = changes.get(start, 0) + 1
        changes[end] = changes.get(end, 0) - 1
    count, prior, overlap, union = 0, None, [], 0
    for time, delta in sorted(changes.items()):
        if prior is not None:
            if count > 0:
                union += time - prior
            if count > 1:
                overlap.append((prior, time))
        count += delta
        prior = time
    return overlap, union


def _combine(paths, origin, limit, operation):
    """Union observed ownership/dependency time, with dependency precedence."""
    observed = [s for path in paths for s in path if s[1] != "unobserved"]
    points = sorted({origin, limit} | {t for s in observed for t in s[2:]})
    result = []
    for start, end in zip(points, points[1:]):
        op, kind = next(((op, kind) for op, kind, a, b in observed
                         if a <= start and end <= b), (operation, "unobserved"))
        if result and result[-1][:2] == (op, kind):
            result[-1] = (op, kind, result[-1][2], end)
        else:
            result.append((op, kind, start, end))
    return result


def _category(row):
    kind = row["event_type"]
    if kind in ("l2_get", "l2_put"):
        return "storage"
    if kind in ("codec_encode", "codec_decode"):
        return "codec"
    if kind in ("kv_import", "kv_export") and row.get("phase") != "restore":
        return "device_transfer_host_window"
    return "serial"


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def critical_path(dag, terminal_event):
    """Return the dependency chain limiting this terminal's observed end.

    Joins select the latest dependency completion (ID breaks ties). A parent
    that encloses its child contributes its prefix before child enqueue/start,
    unioned with the completion chain. Missing queue observations stay unknown.
    The bound permits perfect inter-stage pipelining while retaining observed
    same-stage occupancy. Only serial/queue/gap segments outside all resource
    windows are additive. Host windows are not a device-bandwidth prediction.
    """
    if terminal_event not in dag.records:
        raise TraceValidationError(
            ValidationReport(False, ("missing_terminal",))
        )
    states, order = _causal_states(dag, terminal_event)
    terminal_end = dag.records[terminal_event]["end_ns"]
    limits = {}
    for op, limit in order:
        limits[op] = max(limits.get(op, limit), limit)
    # Report observed host overlap through the terminal, independently of
    # which ownership prefix supplied critical work. Future completions were
    # excluded above, rather than converted to negative/zero-length spans.
    bounded = [dict(dag.records[op], end_ns=min(dag.records[op]["end_ns"],
                                             terminal_end))
               for op in limits if dag.records[op]["start_ns"] < terminal_end]
    overlaps, union = _overlap_intervals(bounded)
    timings, paths, weights, origins = {}, {}, {}, {}
    for state in order:
        op, limit = state
        row = dag.records[op]
        start, end, enqueue = (
            row.get(n) for n in ("start_ns", "end_ns", "enqueue_ns")
        )
        begin = _begin(row)
        timings[op] = OperationTiming(
            None if enqueue is None else start - enqueue,
            end - start,
            sum(max(0, min(end, b) - max(start, a)) for a, b in overlaps),
        )
        dependencies, owner = states[state]
        candidates = []
        if dependencies:
            controlling = max(dependencies, key=lambda p: (p[1], p[0]))
            candidates.append(paths[controlling])
            ready = controlling[1]
        else:
            ready = begin
        if owner:
            candidates.append(paths[owner])
        origins[state] = min([begin] + [origins[p] for p in dependencies]
                             + ([origins[owner]] if owner else []))
        # Slack preserves the observed edge lag, including queue/idle time.
        weights[state] = limit - ready
        own = []
        if enqueue is not None and max(ready, enqueue) < min(start, limit):
            own.append((op, "queue", max(ready, enqueue), min(start, limit)))
        if max(ready, start) < limit:
            own.append((op, "active", max(ready, start), limit))
        paths[state] = _combine(candidates + [own], origins[state], limit, op)
    terminal = (terminal_event, terminal_end)
    origin, segments = origins[terminal], paths[terminal]
    components = dict.fromkeys(
        ("storage", "codec", "device_transfer_host_window", "serial"), 0
    )
    amounts = dict.fromkeys(("active", "queue", "unobserved"), 0)
    resource_windows = []
    for category in components:
        if category == "serial":
            continue
        windows = _merged((r["start_ns"], r["end_ns"])
                          for r in bounded if _category(r) == category)
        components[category] = sum(end - start for start, end in windows)
        resource_windows.extend(windows)
    resource_windows = _merged(resource_windows)
    for op, kind, start, end in segments:
        duration = end - start
        amounts[kind] += duration
        if kind != "unobserved":
            timings[op].critical_ns += duration
        category = _category(dag.records[op]) if kind == "active" else "serial"
        if category == "serial":
            components["serial"] += duration - sum(
                max(0, min(end, b) - max(start, a))
                for a, b in resource_windows
            )
    deadlines = {terminal: terminal_end}
    for state in reversed(order):
        if state not in deadlines:
            continue
        op, limit = state
        if limit == dag.records[op]["end_ns"]:
            timings[op].slack_ns = deadlines[state] - limit
        predecessors, owner = states[state]
        for dep in predecessors:
            deadline = deadlines[state] - weights[state]
            deadlines[dep] = min(deadlines.get(dep, deadline), deadline)
        if owner:
            deadline = deadlines[state] - (limit - owner[1])
            deadlines[owner] = min(deadlines.get(owner, deadline), deadline)
    return CriticalPath(
        terminal_event,
        terminal_end - origin,
        amounts["queue"]
        if all(t.queue_ns is not None for t in timings.values())
        else None,
        amounts["queue"],
        amounts["active"],
        sum(r["end_ns"] - r["start_ns"] for r in bounded) - union,
        amounts["unobserved"],
        max(components[n] for n in components if n != "serial")
        + components["serial"],
        components,
        timings,
    )


TRACE_COLUMNS = tuple(f.name for f in fields(TraceEvent)) + (
    "queue_ns",
    "active_ns",
    "overlap_ns",
    "critical_ns",
    "slack_ns",
)
REQUEST_COLUMNS = (
    "experiment_id",
    "run_id",
    "request_id",
    "terminal_event",
    "total_ns",
    "queue_ns",
    "observed_queue_ns",
    "active_ns",
    "overlap_ns",
    "unobserved_ns",
    "theoretical_lower_bound_ns",
)
BLOCK_COLUMNS = (
    "experiment_id",
    "run_id",
    "request_id",
    "block_key",
    "generation",
    "cache_group",
    "cache_spec",
    "layer_observed",
    "layer",
    "operation_id",
    "event_type",
    "result",
    "raw_bytes",
    "envelope_bytes",
    "connector_read_bytes",
    "connector_write_bytes",
    "active_ns",
    "critical_ns",
)


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _status(path):
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("valid")) is not bool
        or type(data.get("dropped_events")) is not int
        or data["dropped_events"] < 0
        or not isinstance(data.get("reasons"), list)
        or any(type(s) is not str for s in data["reasons"])
    ):
        raise TraceValidationError(ValidationReport(False, ("run_status",)))
    if data.get("export_error"):
        raise TraceValidationError(ValidationReport(False, ("export_error",)))
    return TraceRunStatus(
        data["valid"], tuple(data["reasons"]), data["dropped_events"]
    )


def _merged_status(paths):
    """One run status for the files of every process of a server."""
    statuses = [_status(path) for path in paths]
    return TraceRunStatus(
        all(s.valid for s in statuses),
        tuple(dict.fromkeys(r for s in statuses for r in s.reasons)),
        sum(s.dropped_events for s in statuses),
    )


def analyze(events, run_status, output_dir, terminal_kinds=None):
    """Write validated results; invalidate stale outputs on failed reruns.

    ``events``/``run_status`` may list one file per process. By default every
    request needs a first/output token terminal; given ``terminal_kinds``
    (e.g. kv_import), only requests with such a terminal are analyzed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results, traces, summaries = [], [], []
    events = events if isinstance(events, list) else [events]
    run_status = run_status if isinstance(run_status, list) else [run_status]
    kinds = tuple(terminal_kinds or ("first_token", "output_token"))
    try:
        if len(run_status) != len(events):
            # Each process publishes its status only on a clean close.
            raise TraceValidationError(
                ValidationReport(False, ("run_status",))
            )
        records = [
            json.loads(line)
            for path in events
            for line in path.read_text().splitlines()
        ]
        if not records:
            raise TraceValidationError(
                ValidationReport(False, ("empty_trace",))
            )
        dag = build_operation_dag(records, _merged_status(run_status))
        terminals = [
            op for op in dag.order if dag.records[op]["event_type"] in kinds
        ]
        def identity(row):
            return tuple(row[n] for n in (
                "experiment_id", "run_id", "request_id"
            ))
        if not terminals or terminal_kinds is None and {
            identity(r) for r in records
        } - {
            identity(dag.records[op]) for op in terminals
        }:
            raise TraceValidationError(
                ValidationReport(False, ("missing_terminal",))
            )
        observed = {}
        for terminal in terminals:
            result = critical_path(dag, terminal)
            results.append(asdict(result))
            summaries.append(dict(dag.records[terminal], **asdict(result)))
            for op, timing in result.by_operation.items():
                # Trace row is per operation. Critical metrics are terminal-
                # specific in JSON; keep CSV null when terminals disagree.
                values = asdict(timing)
                if op in observed:
                    values = {
                        k: v if observed[op][k] == v else None
                        for k, v in values.items()
                    }
                observed[op] = values
        traces = [
            dict(dag.records[op], **observed.get(op, {})) for op in dag.order
        ]
        report = dag.validation
    except TraceValidationError as error:
        report = error.report
    except (OSError, ValueError, TypeError) as error:
        report = ValidationReport(
            False, ("input_error:" + type(error).__name__,)
        )
    (output_dir / "validation.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n"
    )
    payload = {
        "valid": report.valid,
        "time_unit": "ns",
        "clock": "host_monotonic",
        "results": results if report.valid else [],
        "lower_bound_model": "observed_stage_occupancy_perfect_pipeline",
        "lower_bound_assumptions": [
            "Preserve same-stage observed interval union; overlap stages fully",
            "Add only serial/queue/gap time outside resource windows",
            "Device transfer uses host windows; device duration is unknown",
        ],
        "device_time_ns": None,
    }
    (output_dir / "critical_path.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    if not report.valid:
        traces, summaries = [], []
    _write_csv(output_dir / "trace.csv", TRACE_COLUMNS, traces)
    _write_csv(output_dir / "request_summary.csv", REQUEST_COLUMNS, summaries)
    _write_csv(
        output_dir / "block_summary.csv",
        BLOCK_COLUMNS,
        [r for r in traces if r.get("block_key") is not None],
    )
    return 0 if report.valid else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--run-status", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--terminal",
        action="append",
        help="event type ending each analyzed path (default: tokens)",
    )
    args = parser.parse_args()
    return analyze(
        args.events, args.run_status, args.output_dir, args.terminal
    )


if __name__ == "__main__":
    raise SystemExit(main())
