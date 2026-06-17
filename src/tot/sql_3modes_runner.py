import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable

import run_sql_3modes as runner


RunFn = Callable[[Any, int, Any, str], dict[str, Any]]


def add_common_search_args(parser) -> None:
    parser.add_argument(
        "--mode",
        type=str,
        default="scheme_b",
        choices=["serial", "layered", "scheme_b", "scheme_h", "dfs"],
    )
    parser.add_argument("--backend", type=str, default="qwen3-32b-vllm")

    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument(
        "--task_ids",
        type=str,
        default="",
        help="Comma-separated 1-based task ids to run. Overrides --start/--end when set.",
    )

    parser.add_argument("--n_propose_sample", type=int, default=6)
    parser.add_argument("--n_select_sample", type=int, default=3)
    parser.add_argument("--n_evaluate_sample", type=int, default=2)

    parser.add_argument("--propose_concurrency", type=int, default=2)
    parser.add_argument("--value_concurrency", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_children_per_parent", type=int, default=8)

    parser.add_argument("--early_stop", type=int, default=1)

    parser.add_argument("--log_interval_s", type=float, default=5.0)
    parser.add_argument("--task_time_limit_s", type=float, default=600.0)
    parser.add_argument("--priority_depth_weight", type=float, default=0.25)
    parser.add_argument("--priority_db_penalty_weight", type=float, default=1.0)

    parser.add_argument("--write_trace", type=int, default=0)
    parser.add_argument("--log_max_events_per_depth", type=int, default=50)

    parser.add_argument("--dfs_value_threshold", type=float, default=5.0)
    parser.add_argument("--dfs_time_limit_s", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=1)


def finalize_common_args(args) -> None:
    args.early_stop = bool(args.early_stop)
    args.write_trace = bool(args.write_trace)


def build_run_idx(task: Any, args: Any, log_dir: str) -> Callable[[int], dict[str, Any]]:
    def _run_idx(idx: int) -> dict[str, Any]:
        if args.mode == "serial":
            return runner._run_serial(task, idx, args, log_dir)
        if args.mode == "layered":
            return runner._run_layered(task, idx, args, log_dir)
        if args.mode == "dfs":
            return runner._run_dfs(task, idx, args, log_dir)
        if args.mode == "scheme_h":
            return runner._run_scheme_h(task, idx, args, log_dir)
        return runner._run_scheme_b(task, idx, args, log_dir)

    return _run_idx


def parse_task_indices(args: Any) -> list[int]:
    if str(getattr(args, "task_ids", "")).strip():
        task_ids = [int(x.strip()) for x in str(args.task_ids).split(",") if x.strip()]
        return [task_id - 1 for task_id in task_ids]
    return [task_id - 1 for task_id in range(int(args.start), int(args.end) + 1)]


def run_indices(indices: list[int], run_idx: Callable[[int], dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    use_task_concurrency = int(getattr(args, "concurrency", 1)) > 1

    if use_task_concurrency:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futs = {ex.submit(run_idx, idx): idx for idx in indices}
            for fut in as_completed(futs):
                idx = futs[fut]
                _consume_result(idx, fut, results)
    else:
        for idx in indices:
            try:
                result = run_idx(idx)
                results.append(result)
                _print_task_done(idx, result)
            except Exception as exc:
                _print_task_failed(idx, exc)

    return sorted(results, key=lambda r: int(r.get("task_id") or 0))


def _consume_result(idx: int, fut, results: list[dict[str, Any]]) -> None:
    try:
        result = fut.result()
        results.append(result)
        _print_task_done(idx, result)
    except Exception as exc:
        _print_task_failed(idx, exc)


def _print_task_done(idx: int, result: dict[str, Any]) -> None:
    task_id = idx + 1
    baseline = result["baseline"]
    step0 = float(result.get("step0_time") or 0.0)
    print(
        f"Task {task_id} done: total={baseline['total_time']:.2f}s "
        f"step0={step0:.2f}s search={baseline['search_time']:.2f}s "
        f"tokens={baseline['tokens']} success={'yes' if baseline['success'] else 'no'}"
    )


def _print_task_failed(idx: int, exc: Exception) -> None:
    task_id = idx + 1
    print(f"Task {task_id} failed: {exc}")
    traceback.print_exc()


def build_log_dir(base_dir: str, dataset: str, run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, dataset, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def write_summary(
    log_dir: str,
    args: Any,
    results: list[dict[str, Any]],
    extra_summary: dict[str, Any] | None = None,
) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": results,
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def collect_optional_stats(task: Any, attr_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr_name in attr_names:
        if not hasattr(task, attr_name):
            continue
        try:
            out[attr_name] = getattr(task, attr_name)()
        except Exception:
            continue
    return out
