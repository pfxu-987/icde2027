
This project is built on top of the Princeton Tree-of-Thought (ToT) codebase and extends it for text-to-SQL experiments on BIRD, Spider, and WTQ.
The current repository mainly keeps our cleaned Bird release branch.

## Environment
Example setup:

The environment setup also follows the original Princeton ToT implementation.
In particular, the package structure and base runtime are inherited from the upstream `tree-of-thought-llm` project, and our code is developed on top of that foundation.

```bash
cd tree-of-thought-llm-bird
conda activate tot
export VLLM_BASE_URL=xxxx
```

Optional environment variables:

```bash
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=qwen3-32b-vllm
```


## Final Bird Command

Your previous final Bird command still works in the cleaned repository structure.
Use:

```bash
cd /home/pengxuemei/tree-of-thought-llm-bird
conda activate tot

export VLLM_BASE_URL=http://127.0.0.1:8005

python run_bird_plan_proxy_3modes.py \
  --mode scheme_b \
  --backend qwen3-32b-vllm \
  --bird_file mini_dev_sqlite.json \
  --bird_db_root src/tot/data/bird/dev_databases \
  --bird_steps 5 \
  --start 1 --end 500 \
  --n_propose_sample 8 \
  --n_select_sample 5 \
  --n_evaluate_sample 3 \
  --propose_concurrency 5 \
  --value_concurrency 24 \
  --batch_size 16 \
  --max_children_per_parent 12 \
  --early_stop 1 \
  --write_trace 1 \
  --log_interval_s 10 \
  --log_max_events_per_depth 50 \
  --task_time_limit_s 600 \
  --priority_depth_weight 0.25 \
  --priority_db_penalty_weight 1.0 \
  --proxy_calibration_mode schema_all
```

