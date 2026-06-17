import os
import time
import requests
import threading
from contextlib import contextmanager
from collections import defaultdict

completion_tokens = prompt_tokens = 0
_USAGE_LOCK = threading.Lock()

# Per-call-type usage buckets, keyed by X-TOT-Call-Type (e.g. propose/value)
usage_by_call_type = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def reset_usage() -> None:
    global completion_tokens, prompt_tokens, usage_by_call_type
    with _USAGE_LOCK:
        completion_tokens = 0
        prompt_tokens = 0
        usage_by_call_type = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def snapshot_usage() -> dict:
    with _USAGE_LOCK:
        return {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
            "by_call_type": {k: dict(v) for k, v in dict(usage_by_call_type).items()},
        }

# API 配置
BASE_URL = os.environ.get("TOT_BASE_URL") or os.environ.get("BASE_URL") or ""
API_KEY = os.environ.get("TOT_API_KEY") or os.environ.get("API_KEY") or ""

# vLLM 本地部署配置
VLLM_BASE_URL = "http://10.123.4.18:8002"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL = "Qwen/Qwen3-32B"

_env_vllm_base_url = os.environ.get("VLLM_BASE_URL")
if _env_vllm_base_url:
    VLLM_BASE_URL = _env_vllm_base_url

_env_vllm_api_key = os.environ.get("VLLM_API_KEY")
if _env_vllm_api_key:
    VLLM_API_KEY = _env_vllm_api_key

_env_vllm_model = os.environ.get("VLLM_MODEL")
if _env_vllm_model:
    VLLM_MODEL = _env_vllm_model

# 支持的模型列表
SUPPORTED_MODELS = {
    'qwen': 'Qwen',
    'llama2-70b': 'llama-2-70b',
    'llama2': 'llama-2-70b',  # 别名
    'llama3-70b': 'llama-3-70b',  # 可用
    'llama3': 'llama-3-70b',  # 别名
    'llama3.3': 'llama-3.3-70b-instruct-fp8-fast',  # 可用
    'qwen3-32b': 'qwen3-32b',  # 可用
    'qwen3-14b': 'qwen3-14b',  # 可用
    'qwen3-8b': 'qwen3-8b',  # 可用
    'qwen3-32b-vllm': 'Qwen/Qwen3-32B',  # vLLM本地部署
}

_REQUEST_EXTRA_HEADERS = None


def set_request_extra_headers(headers: dict | None):
    global _REQUEST_EXTRA_HEADERS
    _REQUEST_EXTRA_HEADERS = headers


def get_request_extra_headers() -> dict | None:
    return _REQUEST_EXTRA_HEADERS


@contextmanager
def request_extra_headers(headers: dict | None):
    old_headers = get_request_extra_headers()
    set_request_extra_headers(headers)
    try:
        yield
    finally:
        set_request_extra_headers(old_headers)


def completions_with_backoff(**kwargs):
    """使用 API 替代 OpenAI API，支持多种模型"""
    model = kwargs.get('model', 'Qwen')
    messages = kwargs.get('messages', [])
    temperature = kwargs.get('temperature', 0.7)
    max_tokens = kwargs.get('max_tokens', 1000)
    n = kwargs.get('n', 1)
    stop = kwargs.get('stop', None)
    extra_headers = kwargs.get('extra_headers', None)

    # 映射模型名称
    model_lower = model.lower()
    if model_lower in SUPPORTED_MODELS:
        model = SUPPORTED_MODELS[model_lower]
    elif model_lower not in ['qwen', 'llama-2-70b', 'llama-3-70b', 'llama-3.3-70b-instruct-fp8-fast']:
        # 如果不是已知模型，默认使用 llama3
        print(f"⚠️ 未知模型 '{model}'，使用默认模型 'llama-3-70b'")
        model = 'llama-3-70b'

    results = []
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    for i in range(n):
        try:
            content, usage = call_qwen_api(
                messages,
                model,
                temperature,
                max_tokens,
                stop,
                extra_headers=extra_headers,
            )
            results.append(content)
            if isinstance(usage, dict):
                usage_prompt += int(usage.get("prompt_tokens") or 0)
                usage_completion += int(usage.get("completion_tokens") or 0)
                if "total_tokens" in usage and usage.get("total_tokens") is not None:
                    usage_total += int(usage.get("total_tokens") or 0)
                else:
                    usage_total += int((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
        except Exception as e:
            print(f"API 调用失败 (第 {i+1}/{n} 次): {e}")
            results.append("")

    # 模拟 OpenAI 响应格式
    class Choice:
        def __init__(self, content):
            self.message = type('obj', (object,), {'content': content})()

    class Usage:
        def __init__(self, prompt_tokens, completion_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class Response:
        def __init__(self, choices, usage):
            self.choices = choices
            self.usage = usage

    # Fallback: 估算 token 数量（当服务端不返回 usage 时）
    prompt_text = ' '.join([m.get('content', '') for m in messages])
    estimated_prompt_tokens = len(prompt_text) // 4
    estimated_completion_tokens = sum(len(r) // 4 for r in results)

    if usage_prompt <= 0 and usage_completion <= 0:
        usage_prompt = int(estimated_prompt_tokens)
        usage_completion = int(estimated_completion_tokens)
        usage_total = int(usage_prompt + usage_completion)

    return Response(
        choices=[Choice(content) for content in results],
        usage=Usage(usage_prompt, usage_completion)
    )


def call_qwen_api(messages, model, temperature, max_tokens, stop=None, extra_headers=None):
    """调用 Qwen API (使用requests库)"""
    global completion_tokens, prompt_tokens

    max_retries = 3
    retry_delay = 2

    # 判断是否使用vLLM本地部署
    use_vllm = model == 'Qwen/Qwen3-32B' or model == 'qwen3-32b-vllm'
    if use_vllm:
        url = f"{VLLM_BASE_URL}/v1/chat/completions"
        api_key = VLLM_API_KEY
        actual_model = VLLM_MODEL
    else:
        url = f"https://{BASE_URL}/v1/chat/completions"
        api_key = API_KEY
        actual_model = model

    if (not use_vllm) and (not api_key or not BASE_URL):
        raise Exception("Missing TOT_API_KEY/TOT_BASE_URL (or API_KEY/BASE_URL). Configure them or use vLLM.")

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            if extra_headers is None:
                extra_headers = get_request_extra_headers()
            if extra_headers:
                headers.update(extra_headers)
            
            payload = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature
            }
            
            # qwen模型需要 chat_template_kwargs 禁用思考模式
            if 'qwen' in model.lower() or 'Qwen' in actual_model:
                payload["chat_template_kwargs"] = {
                    "enable_thinking": False
                }
            
            # vLLM部署需要max_tokens参数
            if use_vllm:
                payload["max_tokens"] = max_tokens
            
            if stop:
                payload["stop"] = stop if isinstance(stop, list) else [stop]

            connect_timeout_s = float(os.getenv("TOT_HTTP_CONNECT_TIMEOUT", "10"))
            if use_vllm:
                read_timeout_s = float(os.getenv("TOT_VLLM_HTTP_READ_TIMEOUT", "180"))
            else:
                read_timeout_s = float(os.getenv("TOT_HTTP_READ_TIMEOUT", "60"))

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout_s, read_timeout_s),
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            # 提取内容
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']

                usage = result.get("usage") if isinstance(result, dict) else None
                if isinstance(usage, dict):
                    p = int(usage.get("prompt_tokens") or 0)
                    c = int(usage.get("completion_tokens") or 0)
                    t = usage.get("total_tokens")
                    t = int(t) if t is not None else int(p + c)
                    usage_norm = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}
                else:
                    # fallback estimate
                    prompt_text = ' '.join([m.get('content', '') for m in messages])
                    p = int(len(prompt_text) // 4)
                    c = int(len(content) // 4)
                    usage_norm = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": int(p + c)}

                return content, usage_norm
            else:
                raise Exception("未获取到任何内容")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ API 调用错误 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            if attempt < max_retries - 1:
                print(f"⏳ 等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"💥 所有重试均失败，返回空字符串")
                return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )


def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens, usage_by_call_type
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        extra_headers = get_request_extra_headers()
        res = completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            extra_headers=extra_headers,
        )
        outputs.extend([choice.message.content for choice in res.choices])

        # log usage once per completions_with_backoff call
        p = int(getattr(res.usage, "prompt_tokens", 0) or 0)
        c = int(getattr(res.usage, "completion_tokens", 0) or 0)
        t = int(p + c)
        ct = None
        if isinstance(extra_headers, dict):
            ct = extra_headers.get("X-TOT-Call-Type")
        with _USAGE_LOCK:
            prompt_tokens += p
            completion_tokens += c
            if ct:
                b = usage_by_call_type[str(ct)]
                b["prompt_tokens"] += p
                b["completion_tokens"] += c
                b["total_tokens"] += t
    return outputs
