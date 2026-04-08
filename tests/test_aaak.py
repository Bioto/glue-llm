"""Tests for AAAK compression helpers."""

import pytest

from gluellm.compression.aaak import (
    AAAKCompressor,
    AAAK_PREAMBLE_MARKER,
    _csv_stats_comment,
    _format_scalar_for_flatten,
    _format_tool_result,
    transcript_from_messages,
)


def test_aaak_encode_tool_round_flattens_json_array_of_objects() -> None:
    """JSON arrays of objects become one readable line per element with key=value pairs."""
    tool_calls = [
        {"id": "h1", "type": "function", "function": {"name": "get_deployments", "arguments": "{}"}},
    ]
    payload = '[{"version":"2.4.0","status":"rollback","rollback_of":"2.4.1"}]'
    tool_messages = [{"role": "tool", "tool_call_id": "h1", "content": payload}]
    out = AAAKCompressor.encode_tool_round(tool_calls, tool_messages, {"h1": "get_deployments"})
    assert "[0]" in out
    assert "version=2.4.0" in out
    assert "status=rollback" in out
    assert "rollback_of=2.4.1" in out
    assert '{"version"' not in out  # not a single-line compact JSON blob


def test_aaak_encode_tool_round_formats_pipe_separated_tools() -> None:
    """Unique-name calls: args are dropped (empty parentheses), results kept."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_files", "arguments": '{"q": "auth"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"p": "auth.py"}'},
        },
    ]
    tool_messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": '["auth.py","config.py"]'},
        {"role": "tool", "tool_call_id": "call_2", "content": "import jwt\n"},
    ]
    id_to_name = {"call_1": "search_files", "call_2": "read_file"}
    out = AAAKCompressor.encode_tool_round(tool_calls, tool_messages, id_to_name)
    assert out.startswith("[AT]\n")
    assert "T:search_files" in out
    # Unique names → args dropped
    assert "q=auth" not in out
    assert "p=auth.py" not in out
    assert "T:search_files()" in out
    assert "T:read_file()" in out
    assert "→" in out
    assert " | " in out
    assert "read_file" in out


def test_aaak_encode_tool_round_keeps_args_for_disambiguation() -> None:
    """Same function called twice in one round: args are kept for both calls."""
    tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_config", "arguments": '{"env": "staging"}'},
        },
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "get_config", "arguments": '{"env": "production"}'},
        },
    ]
    tool_messages = [
        {"role": "tool", "tool_call_id": "c1", "content": "max_connections: 50\nreplicas: 2\n"},
        {"role": "tool", "tool_call_id": "c2", "content": "max_connections: 500\nreplicas: 12\n"},
    ]
    id_to_name = {"c1": "get_config", "c2": "get_config"}
    out = AAAKCompressor.encode_tool_round(tool_calls, tool_messages, id_to_name)
    # Both calls disambiguated by their args; YAML-like results stay multiline (indented)
    assert "env=staging" in out
    assert "env=production" in out
    assert "max_connections: 50" in out
    assert "max_connections: 500" in out


def test_aaak_encode_tool_round_preserves_newlines_for_csv() -> None:
    """CSV tool results keep real newlines (not \\n) for row-wise scanning."""
    tool_calls = [
        {"id": "csv1", "type": "function", "function": {"name": "export_metrics", "arguments": "{}"}},
    ]
    csv_body = "timestamp,service,rps\n2025-04-01T10:00:00Z,auth,1200\n2025-04-01T10:00:00Z,gateway,3400\n"
    tool_messages = [{"role": "tool", "tool_call_id": "csv1", "content": csv_body}]
    out = AAAKCompressor.encode_tool_round(tool_calls, tool_messages, {"csv1": "export_metrics"})
    assert "[AT]\n" in out
    assert "T:export_metrics()" in out
    # Newlines preserved inside the result (not escaped to backslash-n)
    assert "\n2025-04-01T10:00:00Z,auth" in out
    assert "\\n2025" not in out


def test_aaak_encode_tool_round_escapes_pipes_and_newlines_in_result() -> None:
    tool_calls = [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ]
    tool_messages = [{"role": "tool", "tool_call_id": "c1", "content": "a|b\nc"}]
    out = AAAKCompressor.encode_tool_round(tool_calls, tool_messages, {"c1": "f"})
    assert "\\|" in out or "a\\|b" in out
    # Multiline preserved: real newline between escaped pipe line and "c"
    assert "\n  c" in out


def test_format_scalar_for_flatten_avoids_scientific_notation() -> None:
    """Small floats must render as decimal, not scientific notation."""
    assert _format_scalar_for_flatten(0.0000312) == "0.0000312"
    assert _format_scalar_for_flatten(0.00285) == "0.00285"
    assert _format_scalar_for_flatten(23.7) == "23.7"
    assert _format_scalar_for_flatten(1234567.89) == "1234567.89"
    assert _format_scalar_for_flatten(1.0) == "1"
    assert _format_scalar_for_flatten(0.0) == "0"
    # No "e" or "E" in any output
    for v in [0.0000312, 0.00285, 23.7, 1234567.89, 1e-10, 9.99e-7]:
        result = _format_scalar_for_flatten(v)
        assert "e" not in result and "E" not in result, f"Scientific notation in {result!r} for {v}"


def test_csv_stats_comment_finds_peak_per_numeric_column() -> None:
    """_csv_stats_comment returns peak value and label for each numeric column."""
    col_names = ["timestamp", "service", "rps", "error_pct", "latency_p50", "latency_p99"]
    data_rows = [
        "2025-04-01T10:00:00Z,auth,1200,0.2,11,85",
        "2025-04-01T10:00:00Z,gateway,3400,0.8,4,38",
        "2025-04-01T10:00:00Z,billing,890,1.2,22,195",
        "2025-04-01T11:00:00Z,auth,1180,0.3,12,88",
        "2025-04-01T11:00:00Z,gateway,3600,1.8,5,42",
        "2025-04-01T11:00:00Z,billing,920,0.9,24,205",
    ]
    result = _csv_stats_comment(col_names, data_rows)
    assert result.startswith("# peak:")
    assert "error_pct=1.8" in result
    assert "gateway" in result
    assert "2025-04-01T11:00:00Z" in result


def test_format_tool_result_appends_peak_annotation_to_csv() -> None:
    """_format_tool_result appends a '# peak:' summary line to CSV tool results."""
    csv = (
        "timestamp,service,rps,error_pct,latency_p50,latency_p99\n"
        "2025-04-01T10:00:00Z,auth,1200,0.2,11,85\n"
        "2025-04-01T10:00:00Z,gateway,3400,0.8,4,38\n"
        "2025-04-01T10:00:00Z,billing,890,1.2,22,195\n"
        "2025-04-01T11:00:00Z,auth,1180,0.3,12,88\n"
        "2025-04-01T11:00:00Z,gateway,3600,1.8,5,42\n"
        "2025-04-01T11:00:00Z,billing,920,0.9,24,205\n"
    )
    result = _format_tool_result(csv)
    assert "# peak:" in result
    assert "error_pct=1.8" in result
    assert "gateway" in result


def test_ensure_preamble_in_system_appends_once() -> None:
    sys_msg: dict = {"role": "system", "content": "You are helpful."}
    AAAKCompressor.ensure_preamble_in_system(sys_msg)
    assert AAAK_PREAMBLE_MARKER in sys_msg["content"]
    first = sys_msg["content"]
    AAAKCompressor.ensure_preamble_in_system(sys_msg)
    assert sys_msg["content"] == first


def test_transcript_from_messages_includes_tool_results() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]
    t = transcript_from_messages(msgs)
    assert "TOOL_RESULT[t1]" in t
    assert "USR: hi" in t or "USER: hi" in t


@pytest.mark.asyncio
async def test_compress_messages_passthrough_preserves_at_blocks_without_llm() -> None:
    """Pipeline path: pre-existing [AT] blocks must not be re-encoded (no LLM call)."""
    at_block = (
        "[AT]\n"
        "T:get_live_metrics(svc=gateway)→"
        '{"rps":3400,"error_pct":0.4,"shard":"us-east-1a"}'
    )
    old_messages = [
        {"role": "user", "content": "Pull metrics."},
        {"role": "assistant", "content": at_block},
    ]
    out = await AAAKCompressor.compress_messages(
        old_messages,
        model="openai:gpt-4o-mini",
        api_key=None,
    )
    assert out == f"USR: Pull metrics.\n{at_block}"
    assert "us-east-1a" in out
    assert "3400" in out
