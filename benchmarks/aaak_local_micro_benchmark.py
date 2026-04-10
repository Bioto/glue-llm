"""Local (no API) micro-benchmarks for AAAK helper paths.

Run:
  uv run python benchmarks/aaak_local_micro_benchmark.py --iterations 500
"""

from __future__ import annotations

import argparse
import time

from gluellm.compression.aaak import AAAKCompressor, transcript_from_messages


def _large_message_list(n_turns: int) -> list[dict]:
    messages: list[dict] = []
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"step={i} config timeout_ms=8500 pool_size=10"})
        messages.append(
            {
                "role": "assistant",
                "content": f"ACK {i}: rate=/api:10r/m@gateway | cookie=HttpOnly,Secure,SameSite=Strict",
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "fetch", "arguments": '{"url":"https://x.test"}'},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true,"items":[1,2,3]}'})
    return messages


def _bench_transcript(messages: list[dict], iterations: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        transcript_from_messages(messages)
    return time.perf_counter() - t0


def _bench_encode_round(iterations: int) -> float:
    tool_calls = [
        {"id": "a", "function": {"name": "f", "arguments": "{}"}},
        {"id": "b", "function": {"name": "f", "arguments": '{"x":1}'}},
    ]
    tool_msgs = [
        {"tool_call_id": "a", "content": '{"v":1}'},
        {"tool_call_id": "b", "content": '{"v":2}'},
    ]
    id_to_name = {"a": "f", "b": "f"}
    t0 = time.perf_counter()
    for _ in range(iterations):
        AAAKCompressor.encode_tool_round(tool_calls, tool_msgs, id_to_name)
    return time.perf_counter() - t0


def _bench_preamble(iterations: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        m = {"role": "system", "content": "You are helpful."}
        AAAKCompressor.ensure_preamble_in_system(m)
    return time.perf_counter() - t0


def main_sync(args: argparse.Namespace) -> None:
    """Synchronous entry: micro-benchmarks do not perform I/O."""
    it = args.iterations
    turns = args.turns
    messages = _large_message_list(turns)

    transcript_from_messages(messages)
    AAAKCompressor.encode_tool_round(
        [{"id": "a", "function": {"name": "f", "arguments": "{}"}}],
        [{"tool_call_id": "a", "content": "1"}],
        {"a": "f"},
    )

    t_tr = _bench_transcript(messages, it)
    t_enc = _bench_encode_round(it)
    t_pre = _bench_preamble(it)

    print(f"  turns={turns}  iterations={it}")
    print(f"  transcript_from_messages: {t_tr*1000:.2f} ms total  ({t_tr/it*1e6:.2f} µs/iter)")
    print(f"  encode_tool_round:        {t_enc*1000:.2f} ms total  ({t_enc/it*1e6:.2f} µs/iter)")
    print(f"  ensure_preamble_in_system:{t_pre*1000:.2f} ms total  ({t_pre/it*1e6:.2f} µs/iter)")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="AAAK local micro-benchmarks (no network).")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--turns", type=int, default=40, help="Synthetic user/assistant pairs before tool round.")
    args = p.parse_args()
    main_sync(args)


if __name__ == "__main__":
    main()
