#!/usr/bin/env python3
"""Single-request smoke test through the concurrent P/D router."""
import argparse
import asyncio
import time

import aiohttp


def make_token_ids(num_tokens: int, seed: int) -> list[int]:
    state = seed
    result = []
    for _ in range(num_tokens):
        state = (1664525 * state + 1013904223) % (2**32)
        result.append(1000 + state % 49000)
    return result


async def test_pd(router_url: str, d_url: str, num_tokens: int, max_tokens: int, seed: int) -> bool:
    prompt_ids = make_token_ids(num_tokens, seed)
    body = {
        "model": "llama3-p",
        "prompt_token_ids": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
        "return_token_ids": True,
        "request_id": f"smoke-{seed}",
    }
    started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        async with session.post(f"{router_url}/v1/completions", json=body) as resp:
            if resp.status != 200:
                print(f"FAIL HTTP {resp.status}: {await resp.text()}")
                return False
            data = await resp.json()
        reference_body = {
            "model": "llama3-d",
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "return_token_ids": True,
        }
        async with session.post(f"{d_url}/v1/completions", json=reference_body) as resp:
            if resp.status != 200:
                print(f"FAIL direct D reference HTTP {resp.status}: {await resp.text()}")
                return False
            reference = await resp.json()

    choices = data.get("choices", [])
    usage = data.get("usage", {})
    output_ids = choices[0].get("token_ids", []) if choices else []
    reference_choices = reference.get("choices", [])
    reference_ids = reference_choices[0].get("token_ids", []) if reference_choices else []
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(
        f"status=200 prompt_tokens={usage.get('prompt_tokens')} "
        f"output_tokens={len(output_ids)} reference_match={output_ids == reference_ids} "
        f"elapsed_ms={elapsed_ms:.1f} "
        f"pd_metrics={data.get('pd_metrics')}"
    )
    if (usage.get("prompt_tokens") != num_tokens or len(output_ids) != max_tokens
            or output_ids != reference_ids):
        print(f"FAIL unexpected usage/output: usage={usage}, ids={output_ids}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--d-url", default="http://127.0.0.1:8200")
    parser.add_argument("--prompt-tokens", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1500)
    args = parser.parse_args()
    print(f"Testing concurrent P/D path: {args.prompt_tokens} prompt tokens")
    ok = asyncio.run(test_pd(
        args.router_url, args.d_url, args.prompt_tokens, args.max_tokens, args.seed
    ))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
