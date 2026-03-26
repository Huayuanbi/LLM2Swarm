"""
scripts/test_concurrent_vlm.py — Verify Ollama concurrent execution.

Fires N simultaneous requests to the edge VLM and measures:
  - Time for all requests to complete (should be close to a single request
    if Ollama is truly running concurrently, not ~N× longer)
  - Whether thinking blocks are present in the raw response (should be absent
    after the /no_think fix)
  - Whether each response parses to valid JSON

Usage:
    conda activate llm2swarm
    python scripts/test_concurrent_vlm.py            # default: 3 concurrent (matches drone count)
    python scripts/test_concurrent_vlm.py --n 5      # custom concurrency
    python scripts/test_concurrent_vlm.py --n 3 --show-raw   # print raw responses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv
load_dotenv()

import config
from openai import AsyncOpenAI

# ── Minimal prompt (text-only, no image) ─────────────────────────────────────

_SYSTEM = """\
You are the onboard AI co-pilot for an autonomous drone.
Return ONLY valid JSON — no markdown, no commentary.
If the current task is fine: {"decision": "continue"}
If a change is needed:      {"decision": "modify", "new_task": "...", \
"new_action": {"action": "hover", "params": {"duration": 5}}, "memory_update": "..."}
Default to continue unless there is a clear reason to change. /no_think"""

_USER = """\
Drone telemetry:
  pos=(10.0, 20.0, 12.0)  vel=(2.0, 0.0, 0.0)  status=executing  task=go_to_waypoint
Teammates: none
Camera: clear sky, no obstacles visible.
Return your JSON decision. /no_think"""


# ── Single request ─────────────────────────────────────────────────────────────

async def single_request(
    client: AsyncOpenAI,
    req_id: int,
    show_raw: bool,
) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=config.EDGE_VLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER},
            ],
            temperature=0.0,
            timeout=60.0,
        )
        elapsed = time.perf_counter() - t0
        raw = resp.choices[0].message.content or ""

        if show_raw:
            print(f"\n── req {req_id} raw ──\n{raw}\n─────────────────")

        # Check for leftover thinking blocks
        has_think = bool(re.search(r"<think>", raw, re.IGNORECASE))

        # Strip thinking blocks and parse JSON
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            decision = data.get("decision", "?")
            valid_json = True
        else:
            decision = "PARSE_ERROR"
            valid_json = False

        return {
            "id":         req_id,
            "elapsed":    elapsed,
            "decision":   decision,
            "valid_json": valid_json,
            "has_think":  has_think,
            "error":      None,
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "id":         req_id,
            "elapsed":    elapsed,
            "decision":   "ERROR",
            "valid_json": False,
            "has_think":  False,
            "error":      str(e),
        }


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(n: int, show_raw: bool) -> None:
    client = AsyncOpenAI(
        api_key=config.EDGE_VLM_API_KEY,
        base_url=config.EDGE_VLM_BASE_URL,
    )

    print(f"Endpoint : {config.EDGE_VLM_BASE_URL}")
    print(f"Model    : {config.EDGE_VLM_MODEL}")
    print(f"Requests : {n} concurrent")
    print()

    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[single_request(client, i + 1, show_raw) for i in range(n)]
    )
    wall_elapsed = time.perf_counter() - wall_start

    # ── Print table ──
    print(f"{'Req':>4}  {'Time (s)':>9}  {'Decision':<12}  {'JSON':>5}  {'<think>':>7}  Error")
    print("─" * 62)
    for r in results:
        think_flag = "YES ⚠" if r["has_think"] else "no"
        json_flag  = "✓" if r["valid_json"] else "✗"
        err        = r["error"] or ""
        print(
            f"{r['id']:>4}  {r['elapsed']:>9.2f}  {r['decision']:<12}  "
            f"{json_flag:>5}  {think_flag:>7}  {err}"
        )

    print("─" * 62)
    print(f"Wall time for all {n} requests: {wall_elapsed:.2f} s")

    # ── Verdict ──
    valid    = [r for r in results if r["valid_json"]]
    thinking = [r for r in results if r["has_think"]]
    errors   = [r for r in results if r["error"]]

    print()
    if errors:
        print(f"  ✗  {len(errors)}/{n} requests failed")
    else:
        print(f"  ✓  All {n} requests succeeded")

    if thinking:
        print(f"  ⚠  {len(thinking)}/{n} responses still contained <think> blocks")
        print("     → /no_think may not be supported; thinking stripped at parse time")
    else:
        print(f"  ✓  No thinking blocks detected — /no_think is working")

    if valid:
        avg_single = sum(r["elapsed"] for r in valid) / len(valid)
        print(f"  ✓  Average single-request time: {avg_single:.2f} s")
        if n > 1:
            concurrency_ratio = wall_elapsed / avg_single
            if concurrency_ratio < 1.5:
                print(f"  ✓  Concurrency ratio: {concurrency_ratio:.2f}× "
                      f"(≈1.0 = fully parallel) — Ollama concurrent mode confirmed")
            else:
                print(f"  ⚠  Concurrency ratio: {concurrency_ratio:.2f}× "
                      f"(expected ≈1.0 if concurrent) — may be running sequentially")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Ollama concurrent VLM execution")
    parser.add_argument("--n",        type=int,  default=3,     help="Number of concurrent requests")
    parser.add_argument("--show-raw", action="store_true",      help="Print raw model responses")
    args = parser.parse_args()
    asyncio.run(run(args.n, args.show_raw))


if __name__ == "__main__":
    main()
