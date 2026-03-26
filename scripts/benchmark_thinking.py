"""
scripts/benchmark_thinking.py — Benchmark qwen3 thinking-mode latency.

Uses streaming so we can measure:
  - Time to first token
  - Time from start to </think> (thinking phase)
  - Time from </think> to end   (answer phase)
  - Total time

Runs N trials for each mode (thinking ON / OFF) and prints a summary.

Usage:
    conda activate llm2swarm
    python scripts/benchmark_thinking.py
    python scripts/benchmark_thinking.py --trials 5 --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv
load_dotenv()

import config
from openai import AsyncOpenAI

# ── Prompt (same as VLMAgent uses) ────────────────────────────────────────────

_SYSTEM = """\
You are the onboard AI co-pilot for a single autonomous drone in a multi-drone swarm.
Return ONLY valid JSON — no markdown, no extra commentary.
If the current task is fine: {"decision": "continue"}
If a change is needed:
{
  "decision": "modify",
  "new_task": "<description>",
  "new_action": {"action": "<skill>", "params": {}},
  "memory_update": "<observation>"
}
Default to continue unless there is a clear reason to change."""

_USER = """\
=== DRONE TELEMETRY ===
ID:           drone_1
Position:     x=42.30  y=38.10  z=12.00  (metres)
Velocity:     vx=2.10  vy=0.80  vz=0.00  (m/s)
Status:       executing
Active task:  go_to_waypoint

=== TEAMMATE STATES ===
  drone_2: pos=(61.0,1.2,10.0) status=executing obs=[none]
  drone_3: pos=(0.0,0.0,8.0)   status=executing obs=[none]

=== CAMERA IMAGE ===
No image available in benchmark mode. Assume clear sky, no obstacles.

Based on this information, return your JSON decision."""


# ── Single streaming request ──────────────────────────────────────────────────

async def timed_request(
    client:   AsyncOpenAI,
    no_think: bool,
    trial_id: str,
) -> dict:
    """
    Stream one request and record timing milestones.
    Returns a dict with timing fields and the full response text.
    """
    user_msg = _USER + ("\n/no_think" if no_think else "")

    t_start = time.perf_counter()
    t_first_token: float | None = None
    t_think_end:   float | None = None

    full_text  = ""
    in_think   = False
    think_done = False

    try:
        stream = await client.chat.completions.create(
            model=config.EDGE_VLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            stream=True,
            timeout=120.0,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            if t_first_token is None:
                t_first_token = time.perf_counter()

            full_text += delta

            # Detect thinking-block boundaries
            if not think_done:
                if "<think>" in full_text:
                    in_think = True
                if in_think and "</think>" in full_text:
                    t_think_end = time.perf_counter()
                    in_think    = False
                    think_done  = True

        t_end = time.perf_counter()

        has_think    = "<think>" in full_text
        think_tokens = ""
        if has_think:
            m = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
            think_tokens = m.group(1).strip() if m else ""

        return {
            "trial":           trial_id,
            "mode":            "no_think" if no_think else "thinking",
            "total_s":         t_end - t_start,
            "first_token_s":   (t_first_token - t_start) if t_first_token else None,
            "think_phase_s":   (t_think_end - t_start) if t_think_end else None,
            "answer_phase_s":  (t_end - t_think_end) if t_think_end else None,
            "has_think":       has_think,
            "think_chars":     len(think_tokens),
            "response":        full_text.strip(),
            "error":           None,
        }

    except Exception as e:
        return {
            "trial":          trial_id,
            "mode":           "no_think" if no_think else "thinking",
            "total_s":        time.perf_counter() - t_start,
            "first_token_s":  None,
            "think_phase_s":  None,
            "answer_phase_s": None,
            "has_think":      False,
            "think_chars":    0,
            "response":       "",
            "error":          str(e),
        }


# ── Run a batch of trials ─────────────────────────────────────────────────────

async def run_trials(
    client:      AsyncOpenAI,
    no_think:    bool,
    trials:      int,
    concurrency: int,
) -> list[dict]:
    """Run `trials` requests, `concurrency` at a time."""
    results = []
    mode_label = "no_think" if no_think else "thinking"
    for batch_start in range(0, trials, concurrency):
        batch_ids = [
            f"{mode_label}-{i+1}"
            for i in range(batch_start, min(batch_start + concurrency, trials))
        ]
        batch = await asyncio.gather(
            *[timed_request(client, no_think, bid) for bid in batch_ids]
        )
        results.extend(batch)
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    avg = sum(values) / len(values)
    mn  = min(values)
    mx  = max(values)
    return f"avg={avg:.2f}s  min={mn:.2f}s  max={mx:.2f}s"


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(trials: int, concurrency: int, show_responses: bool) -> None:
    client = AsyncOpenAI(
        api_key=config.EDGE_VLM_API_KEY,
        base_url=config.EDGE_VLM_BASE_URL,
    )

    print(f"Model      : {config.EDGE_VLM_MODEL}")
    print(f"Endpoint   : {config.EDGE_VLM_BASE_URL}")
    print(f"Trials     : {trials} per mode  |  Concurrency: {concurrency}")
    print()

    # ── Thinking ON ──
    print(f"── Mode: THINKING ON ({trials} trials) ──────────────────────────")
    think_results = await run_trials(client, no_think=False, trials=trials, concurrency=concurrency)

    # ── Thinking OFF ──
    print(f"── Mode: THINKING OFF ({trials} trials) ─────────────────────────")
    nothink_results = await run_trials(client, no_think=True, trials=trials, concurrency=concurrency)

    # ── Per-trial table ──
    print()
    print(f"{'Trial':<18} {'Mode':<12} {'Total':>8} {'→<think>':>10} {'think dur':>10} "
          f"{'answer':>8} {'think_chars':>12}  Response snippet")
    print("─" * 110)

    for r in think_results + nothink_results:
        if r["error"]:
            print(f"  {r['trial']:<16} {r['mode']:<12}  ERROR: {r['error']}")
            continue
        think_s   = f"{r['think_phase_s']:.2f}s" if r["think_phase_s"] else "—"
        answer_s  = f"{r['answer_phase_s']:.2f}s" if r["answer_phase_s"] else "—"
        first_s   = f"{r['first_token_s']:.2f}s" if r["first_token_s"] else "—"
        snippet   = r["response"].replace("\n", " ")[:60]
        # Strip think block from snippet for readability
        snippet   = re.sub(r"<think>.*?</think>", "[<think>]", snippet, flags=re.DOTALL)
        print(
            f"  {r['trial']:<16} {r['mode']:<12} {r['total_s']:>7.2f}s "
            f"{first_s:>10} {think_s:>10} {answer_s:>8} {r['think_chars']:>12}  {snippet}"
        )

    # ── Summary ──
    print()
    print("═" * 70)
    print("SUMMARY")
    print("═" * 70)

    for label, results in [("THINKING ON", think_results), ("THINKING OFF", nothink_results)]:
        ok = [r for r in results if not r["error"]]
        if not ok:
            print(f"\n  {label}: all requests failed")
            continue

        total_times   = [r["total_s"]        for r in ok]
        think_times   = [r["think_phase_s"]  for r in ok if r["think_phase_s"]]
        answer_times  = [r["answer_phase_s"] for r in ok if r["answer_phase_s"]]
        think_chars   = [r["think_chars"]    for r in ok if r["think_chars"] > 0]

        print(f"\n  {label}  ({len(ok)}/{len(results)} succeeded)")
        print(f"    Total time   : {stats(total_times)}")
        if think_times:
            print(f"    Think phase  : {stats(think_times)}")
            print(f"    Answer phase : {stats(answer_times)}")
            avg_chars = sum(think_chars) / len(think_chars) if think_chars else 0
            print(f"    Think tokens : avg {avg_chars:.0f} chars in <think> block")
        else:
            print(f"    (no <think> blocks detected)")

    # ── Overhead ──
    think_ok   = [r for r in think_results  if not r["error"]]
    nothink_ok = [r for r in nothink_results if not r["error"]]
    if think_ok and nothink_ok:
        avg_think   = sum(r["total_s"] for r in think_ok)   / len(think_ok)
        avg_nothink = sum(r["total_s"] for r in nothink_ok) / len(nothink_ok)
        overhead    = avg_think - avg_nothink
        overhead_pct = overhead / avg_nothink * 100 if avg_nothink else 0
        print()
        print(f"  Thinking overhead: +{overhead:.2f}s per request ({overhead_pct:.0f}% slower)")
        print(f"  VLM_TIMEOUT recommendation: {avg_think * 1.5:.0f}s "
              f"(1.5× avg with thinking)")

    # ── Optional: show full responses ──
    if show_responses:
        print("\n\n── Full responses ──────────────────────────────────────────────")
        for r in think_results + nothink_results:
            print(f"\n[{r['trial']}]")
            print(r["response"] or r["error"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark qwen3 thinking-mode latency"
    )
    parser.add_argument("--trials",       type=int,  default=3,
                        help="Number of trials per mode (default: 3)")
    parser.add_argument("--concurrency",  type=int,  default=1,
                        help="Requests per batch (default: 1 = sequential)")
    parser.add_argument("--show-responses", action="store_true",
                        help="Print full model responses at the end")
    args = parser.parse_args()
    asyncio.run(run(args.trials, args.concurrency, args.show_responses))


if __name__ == "__main__":
    main()
