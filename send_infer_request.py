#!/usr/bin/env python3
"""
Send a single inference request using the existing DataLoader and curl.
Usage examples:
  python send_infer_request.py \
    --data-dir /home/ecnu/disk/wzq/moe-inference/data/wikitext-103-v1 \
    --branch test \
    --host 127.0.0.1 --port 30000 \
    --max-new-tokens 128 --temperature 0.8

If curl is unavailable or you prefer pure Python, add --use-requests to send via requests.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MOE_ROOT = os.path.join(ROOT, "moe-inference")

if MOE_ROOT not in sys.path:
    sys.path.insert(0, MOE_ROOT)

try:
    from data.dataloader import DataLoader  # type: ignore
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"[ERROR] cannot import DataLoader: {e}\n")
    sys.exit(1)


def build_payload(
    data_dir: str,
    branch: str,
    batch_size: Optional[int],
    data_idx: Optional[List[int]],
    max_new_tokens: int,
    temperature: float,
    model_path: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
) -> str:
    loader = DataLoader(data_dir, branch)
    tokenizer = None
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        loader.tokenizer = tokenizer
    if max_input_tokens is not None:
        loader.set_max_length(max_input_tokens)
    if batch_size is not None:
        texts = loader.getRandomBatch(batch_size)
    elif data_idx is not None:
        texts = loader.getBatch(data_idx)

    answers = None
    if len(texts)>0 and isinstance(texts[0], tuple):
        answers = [a for _, a in texts]
        texts = [t for t, _ in texts]

    # 单条时发字符串，多条时发列表，供后端做批处理
    payload = {
        "text": texts if len(texts) > 1 else texts[0],
        "sampling_params": {
            "max_new_tokens": max_new_tokens, # 这里就是你传进来的 1024
            "temperature": temperature,
        },
        # 为了兼容性，有些版本也可以在外面保留一份
        "max_new_tokens": max_new_tokens,
    }
    return json.dumps(payload, ensure_ascii=False), answers 


def send_via_curl(host: str, port: int, payload: str, answers: Optional[List[str]] = None, output: Optional[str] = None) -> int:
    url = f"http://{host}:{port}/generate"
    cmd = [
        "curl",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        payload,
    ]
    return subprocess.call(cmd)


def send_via_requests(host: str, port: int, payload: str, answers: Optional[List[str]] = None, output: Optional[str] = None) -> int:
    try:
        import requests  # type: ignore
    except Exception:
        sys.stderr.write("[ERROR] requests is not installed; use curl or install requests\n")
        return 1
    url = f"http://{host}:{port}/generate"
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, data=payload.encode("utf-8"), headers=headers, timeout=360)
    sys.stdout.write(resp.text + "\n")

    if resp.ok:
        data = resp.json()
        
        # 如果指定了输出文件，保存结果
        if output:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output), exist_ok=True)
            
            # 批量推理：每个结果对应一个answer
            if isinstance(data, list):
                with open(output, "a") as f:
                    for idx, item in enumerate(data):
                        generated_text = item.get("text", "")
                        answer = answers[idx] if answers and idx < len(answers) else "N/A"
                        json.dump({"text": generated_text, "answer": answer}, f)
                        f.write("\n")
            # 单条推理
            else:
                generated_text = data.get("text", "")
                answer = answers[0] if answers and len(answers) > 0 else "N/A"
                with open(output, "a") as f:
                    json.dump({"text": generated_text, "answer": answer}, f)
                    f.write("\n")
    return 0 if resp.ok else 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "moe-inference", "data", "wikitext-103-v1"))
    p.add_argument("--branch", default="test")
    p.add_argument("--model-path", default=None, help="path or repo id for tokenizer loading")
    p.add_argument("--batch-size", type=int, default=None, help="number of samples to send in one request")
    p.add_argument("--data-idx", type=int, nargs="+", default=None, help="specific data indices to send")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-input-tokens", type=int, default=None, help="truncate inputs to this token length via tokenizer")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--use-requests", action="store_true", help="send via requests instead of curl")
    p.add_argument("--output", default="/home/ecnu/disk/wzq/output/default_results.json", help="output file path to save results and answers")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload, answers = build_payload(
        args.data_dir,
        args.branch,
        args.batch_size,
        args.data_idx,
        args.max_new_tokens,
        args.temperature,
        args.model_path,
        args.max_input_tokens,
    )
    sys.stderr.write(f"[INFO] payload from {args.data_dir}/{args.branch}: {payload[:120]}\n")
    sys.stderr.write(f"[INFO] output will be saved to: {args.output}\n")
    
    if args.use_requests:
        return send_via_requests(args.host, args.port, payload, answers, args.output)
    return send_via_curl(args.host, args.port, payload, answers, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
