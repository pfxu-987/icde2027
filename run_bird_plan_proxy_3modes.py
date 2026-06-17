import os
import sys

sys.path.insert(0, "src")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")

import argparse

import tot.models as models
from tot.sql_3modes_runner import (
    add_common_search_args,
    build_log_dir,
    build_run_idx,
    collect_optional_stats,
    finalize_common_args,
    parse_task_indices,
    run_indices,
    write_summary,
)
from tot.tasks.bird_plan_proxy import BirdPlanProxyTask

_env_vllm_base_url = os.environ.get("VLLM_BASE_URL")
if _env_vllm_base_url:
    models.VLLM_BASE_URL = _env_vllm_base_url

_env_vllm_api_key = os.environ.get("VLLM_API_KEY")
if _env_vllm_api_key:
    models.VLLM_API_KEY = _env_vllm_api_key

_env_vllm_model = os.environ.get("VLLM_MODEL")
if _env_vllm_model:
    models.VLLM_MODEL = _env_vllm_model


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_search_args(parser)
    parser.add_argument("--bird_file", type=str, default="mini_dev_sqlite.json")
    parser.add_argument("--bird_db_root", type=str, default="src/tot/data/bird/dev_databases")
    parser.add_argument("--bird_steps", type=int, default=5)
    parser.add_argument("--bird_max_schema_chars", type=int, default=12000)
    parser.add_argument("--disable_cache", type=int, default=0)

    parser.add_argument("--decomp_rounds", type=int, default=2)
    parser.add_argument("--decomp_seed_sample", type=int, default=1)
    parser.add_argument("--decomp_repair_sample", type=int, default=4)
    parser.add_argument("--decomp_seed_temperature", type=float, default=0.0)
    parser.add_argument("--decomp_repair_temperature", type=float, default=0.7)
    parser.add_argument("--decomp_log_top_k", type=int, default=10)

    parser.add_argument("--proxy_calibration_mode", type=str, default="schema_all", choices=["schema_all", "gold_sql"])
    parser.add_argument("--proxy_calibration_start", type=int, default=1)
    parser.add_argument("--proxy_calibration_end", type=int, default=50)
    parser.add_argument("--skip_proxy_calibration", type=int, default=0)

    args = parser.parse_args()
    finalize_common_args(args)
    args.skip_proxy_calibration = bool(args.skip_proxy_calibration)
    args.disable_cache = bool(args.disable_cache)

    task = BirdPlanProxyTask(
        file=args.bird_file,
        db_root=args.bird_db_root,
        steps=int(args.bird_steps),
        max_schema_chars=int(args.bird_max_schema_chars),
    )
    task.disable_cache = bool(args.disable_cache)

    calibration_stats = None
    if not args.skip_proxy_calibration:
        if args.proxy_calibration_mode == "schema_all":
            calibration_stats = task.calibrate_proxy_from_schema()
        else:
            cal_start_idx = max(0, int(args.proxy_calibration_start) - 1)
            cal_end_idx = max(cal_start_idx, int(args.proxy_calibration_end))
            calibration_stats = task.calibrate_proxy(cal_start_idx, cal_end_idx)

    cache_suffix = "_no_cache" if args.disable_cache else ""
    log_dir = build_log_dir("logs", "bird", f"{args.mode}_decomp_repair_plan_proxy{cache_suffix}")

    print(f"logs_dir: {log_dir}")
    if calibration_stats is not None:
        print(
            "proxy_calibration: "
            f"mode={args.proxy_calibration_mode} "
            f"dbs={calibration_stats.get('n_calibrated_dbs', 0)} "
            f"global_bmax={float(calibration_stats.get('global_bmax') or 0.0):.4f}"
        )

    results = run_indices(parse_task_indices(args), build_run_idx(task, args, log_dir), args)
    summary = {}
    if calibration_stats is not None:
        summary["proxy_calibration_stats"] = calibration_stats
    summary.update(
        collect_optional_stats(
            task,
            ["semantic_cache_stats", "execution_cache_stats", "proxy_cost_stats"],
        )
    )
    write_summary(log_dir, args, results, summary)


if __name__ == "__main__":
    main()
