"""AAAK — lossless shorthand dialect for compressing agent context.

Structured text any LLM can read without a separate decoder. Used for
conversation compression and optional tool-round condensing in GlueLLM.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# Grammar reference injected into compression prompts and system preambles.
AAAK_SPEC = """AAAK lossless shorthand — rules:
ROLES: USR: AST: T:name()→val | T:name(key=val)→val (args only when disambiguating)
LISTS: items in [] brackets: [a,b,c] | ordered steps: 1→2→3→4
ATTRS: keep security/config attrs verbatim: HttpOnly,Secure,SameSite=Strict
SCHEMA: col:purpose notation: replaced_by:replay_detect
CONFIG: key=val pairs: timeout_ms=8500 | pool_size=10
NUMBERS: preserve exact values, units: 847errors | 15min | 8500ms
FACTS: separate with | ; chain with → ; relate with .
BREVITY: 3-4 char handles for names only, never for config keys or values"""

AAAK_PREAMBLE_MARKER = "[AAAK decoding hint]"

# Short decode hint for system messages (full AAAK_SPEC lives in compression prompt only).
_AAAK_DECODE_HINT = (
    "Decode [AAAK CTX] and [AT]: USR:/AST: turns; T:name()→result for tools (args only when same name used twice).\n"
    "Lists [a,b,c]; ordered steps 1→2→3; config key=val; schema col:purpose; "
    "security attrs verbatim (HttpOnly,Secure,SameSite=Strict)."
)

_COMPRESS_SYSTEM = (
    "You are an expert lossless compressor. Convert the transcript into AAAK shorthand. "
    "You MUST recover every technical fact exactly — especially rate-limit layers, "
    "schema column purposes, cookie flags, and numbered steps.\n\n"
    + AAAK_SPEC
    + "\n\nWhen you see [AT] blocks, first decode them (T:name()→val) then re-encode "
      "the extracted facts using the same rules. Output ONLY the AAAK-encoded lines "
      "(no markdown fences, no preamble). Every fact from the transcript must be "
      "recoverable from your encoding. No explanations."
)

_COMPRESS_USER_PREFIX = (
    "Encode in AAAK. MUST preserve ALL of these EXACTLY (if present):\n"
    "1. Exact numbers with units: 15min, 8500ms, 900 (expires_in), 847errors\n"
    "2. Config keys=values verbatim: timeout_ms=8500, pool_size=10, JWT_SECRET_KEY\n"
    "3. Rate limits WITH layer: 10 req/min at gateway/nginx, 1000 req/hour at app\n"
    "4. Security/cookie attributes verbatim: HttpOnly,Secure,SameSite=Strict\n"
    "5. Schema columns with purposes: replaced_by:replay_detect\n"
    "6. Ordered steps exactly as 1→2→3→4→5 (logout flow)\n"
    "7. Algorithm names: HS256, PBKDF2-SHA256\n"
    "8. Facts inside [AT] blocks — decode T:name()→val first, then re-encode every number, "
    "config key=val, and schema fact exactly as written.\n\n"
    "Few-shot (style guide, not literal copy):\n"
    "Input: rate limit on /auth/login is 10 req/min at gateway layer\n"
    "Output: rate=10r/m@gateway\n\n"
    "Input: cookie should be HttpOnly,Secure,SameSite=Strict\n"
    "Output: cookie=HttpOnly,Secure,SameSite=Strict\n\n"
    "Input: logout: DELETE /auth/session, SHA-256 hash token, UPDATE revoked_at, 204, clear cookie\n"
    "Output: logout:1→DELETE /auth/session;2→SHA256 hash;3→UPDATE revoked_at;4→204;5→clear cookie\n\n"
    "NEVER summarize or drop these. Output ONLY AAAK lines.\n"
)


def _message_content_to_text(msg: dict[str, Any]) -> str:
    """Best-effort string content from an API-style message dict."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return " ".join(text_parts)
    return str(content)


def transcript_from_messages(messages: list[dict[str, Any]]) -> str:
    """Build a readable transcript for compression (user/assistant/tool text)."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "tool":
            tc_id = msg.get("tool_call_id", "")
            body = _message_content_to_text(msg)
            lines.append(f"TOOL_RESULT[{tc_id}]: {body}")
            continue
        if role == "assistant" and msg.get("tool_calls"):
            tc_parts: list[str] = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name", "?") if isinstance(fn, dict) else getattr(fn, "name", "?")
                args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                tc_parts.append(f"{name}({args})")
            content = _message_content_to_text(msg)
            suffix = f" | calls: {', '.join(tc_parts)}" if tc_parts else ""
            lines.append(f"ASSISTANT:{suffix}\n{content}".strip())
            continue
        content = _message_content_to_text(msg)
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _escape_aaak_value(value: str, max_len: int = 400) -> str:
    """Single-line escape for tool results and inline values."""
    s = str(value).replace("\n", "\\n").replace("\r", "\\r")
    s = s.replace("|", "\\|")
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_scalar_for_flatten(v: Any) -> str:
    """Render a JSON scalar (or value inlined on one line) for flattened AAAK lines."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, str):
        s = v.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("|", "\\|")
        if not s or any(ch in s for ch in ' \t="\''):
            return '"' + s.replace('"', '\\"') + '"'
        return s
    return _format_scalar_for_flatten(str(v))


def _flatten_dict_lines(d: dict[str, Any], prefix: str) -> list[str]:
    """Flatten a JSON object to key.path=value lines (sorted keys for determinism)."""
    if not d:
        return [f"{prefix}=<empty>" if prefix else "<empty>"]
    lines: list[str] = []
    for k, v in sorted(d.items()):
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            lines.extend(_flatten_dict_lines(v, p))
        elif isinstance(v, list):
            lines.extend(_flatten_list_lines(v, p))
        else:
            lines.append(f"{p}={_format_scalar_for_flatten(v)}")
    return lines


def _flatten_list_lines(arr: list[Any], prefix: str) -> list[str]:
    """Flatten a JSON array: one line per element; dict elements use ``[i] k=v ...``."""
    if not arr:
        return [f"{prefix}=[]" if prefix else "<empty>"]
    lines: list[str] = []
    for i, item in enumerate(arr):
        label = f"{prefix}[{i}]" if prefix else f"[{i}]"
        if isinstance(item, dict):
            scalar_parts: list[str] = []
            nested: list[str] = []
            for k, v in sorted(item.items()):
                if isinstance(v, dict):
                    nested.extend(_flatten_dict_lines(v, f"{label}.{k}"))
                elif isinstance(v, list):
                    nested.extend(_flatten_list_lines(v, f"{label}.{k}"))
                else:
                    scalar_parts.append(f"{k}={_format_scalar_for_flatten(v)}")
            if scalar_parts:
                lines.append(f"{label} " + " ".join(scalar_parts))
            elif not nested:
                lines.append(f"{label} <empty>")
            lines.extend(nested)
        elif isinstance(item, list):
            lines.extend(_flatten_list_lines(item, label))
        else:
            lines.append(f"{label}={_format_scalar_for_flatten(item)}")
    return lines


def _flatten_json_to_body(obj: Any) -> str:
    """Turn parsed JSON into indented multi-line flattened text (leading newline)."""
    if isinstance(obj, dict):
        inner = _flatten_dict_lines(obj, "")
    elif isinstance(obj, list):
        inner = _flatten_list_lines(obj, "")
    else:
        inner = [_format_scalar_for_flatten(obj)]
    return "\n" + "\n".join(f"  {ln}" for ln in inner)


def _escape_pipe_in_multiline_line(line: str) -> str:
    """Escape ``|`` so tool-round `` | `` delimiters are not ambiguous."""
    return line.replace("|", "\\|")


def _format_multiline_preserved(s: str) -> str:
    """Keep original newlines; indent each line; escape pipes per line."""
    lines = s.split("\n")
    body = "\n".join(f"  {_escape_pipe_in_multiline_line(ln)}" for ln in lines)
    return "\n" + body


def _csv_stats_comment(col_names: list[str], data_rows: list[str]) -> str:
    """Return a '# peak: col=val(labels)' comment for each numeric column.

    For each column that is entirely numeric, records the max value and the
    corresponding label-column values from that row. This lets a model recall
    cross-row aggregate facts (e.g. peak error_pct) without scanning all rows.
    """
    rows = [r.split(",") for r in data_rows if r.strip()]
    if not rows:
        return ""
    n = len(col_names)
    rows = [r for r in rows if len(r) == n]
    numeric_cols: list[int] = []
    for ci in range(n):
        try:
            [float(r[ci]) for r in rows]
            numeric_cols.append(ci)
        except ValueError:
            pass
    label_cols = [ci for ci in range(n) if ci not in numeric_cols]
    if not numeric_cols:
        return ""
    parts: list[str] = []
    for ci in numeric_cols:
        vals = [float(r[ci]) for r in rows]
        peak_val = max(vals)
        peak_row = next(rows[i] for i, v in enumerate(vals) if v == peak_val)
        labels = ",".join(peak_row[lc] for lc in label_cols)
        parts.append(f"{col_names[ci]}={peak_val:g}({labels})")
    return "# peak: " + " ".join(parts)


def _format_tool_result(raw: str, *, max_len: int = 2000) -> str:
    """Format tool result for AAAK: flatten JSON to readable lines; preserve CSV / prose newlines."""
    s = str(raw).strip()

    if s.startswith("{") or s.startswith("["):
        try:
            obj = json.loads(s)
            flat = _flatten_json_to_body(obj)
            if len(flat) > max_len:
                return flat[: max_len - 3] + "..."
            return flat
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if "\n" in s:
        # CSV: keep real newlines so models can scan rows (e.g. peak in a column).
        lines = s.split("\n")
        header = lines[0].strip()
        if header and "," in header and ":" not in header:
            comma_count = header.count(",")
            data_rows = [
                ln
                for ln in lines[1:]
                if ln.strip() and ln.strip().count(",") == comma_count
            ]
            if len(data_rows) >= 2:
                stats = _csv_stats_comment(header.split(","), data_rows)
                annotated = s + ("\n" + stats if stats else "")
                if len(annotated) > max_len:
                    return annotated[: max_len - 3] + "..."
                return annotated

        preserved = _format_multiline_preserved(s)
        if len(preserved) > max_len:
            return preserved[: max_len - 3] + "..."
        return preserved

    return _escape_aaak_value(s, max_len=max_len)


def _format_tool_args(args_str: str) -> str:
    """Format tool call arguments for AAAK: JSON object to key=val pairs joined with ;."""
    s = str(args_str).strip()
    if s in ("{}", ""):
        return ""
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _escape_aaak_value(s, max_len=400)
    if not isinstance(obj, dict):
        return _escape_aaak_value(s, max_len=400)
    pairs: list[str] = []
    for k, v in obj.items():
        if isinstance(v, list):
            val = "[" + ",".join(str(i) for i in v) + "]"
        elif isinstance(v, dict):
            val = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
        elif isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = "null"
        else:
            val = str(v)
        pairs.append(f"{k}={val}")
    return ";".join(pairs)


class AAAKCompressor:
    """Encode conversation history and tool rounds into AAAK shorthand."""

    @staticmethod
    def get_spec_preamble() -> str:
        """Short reminder for the main model so it can read [AAAK CTX] blocks."""
        return f"{AAAK_PREAMBLE_MARKER}\n{_AAAK_DECODE_HINT}"

    @staticmethod
    def ensure_preamble_in_system(system_message: dict[str, Any]) -> None:
        """Append the decoding hint to a system message once (mutates dict)."""
        content = system_message.get("content") or ""
        if AAAK_PREAMBLE_MARKER in content:
            return
        system_message["content"] = content.rstrip() + "\n\n" + AAAKCompressor.get_spec_preamble()

    @staticmethod
    def encode_tool_round(
        tool_calls: list[dict[str, Any]],
        tool_messages: list[dict[str, Any]],
        id_to_name: dict[str, str],
    ) -> str:
        """Deterministic AAAK encoding for one tool round (no LLM call).

        Args are included only when the same function name appears more than once
        in the round (for disambiguation). Otherwise the parentheses are empty.
        """
        args_by_id: dict[str, str] = {}
        for tc in tool_calls:
            tc_id = tc.get("id", "") or ""
            fn = tc.get("function") or {}
            if isinstance(fn, dict):
                args_by_id[tc_id] = str(fn.get("arguments") or "{}")
            else:
                args_by_id[tc_id] = str(getattr(fn, "arguments", "{}") or "{}")

        # Count how many times each function name appears in this round
        name_counts: Counter[str] = Counter(id_to_name.values())

        segments: list[str] = []
        for tool_msg in tool_messages:
            tc_id = tool_msg.get("tool_call_id", "") or ""
            name = id_to_name.get(tc_id, tc_id or "unknown")
            args = args_by_id.get(tc_id, "{}")
            raw = tool_msg.get("content", "")
            if not isinstance(raw, str):
                try:
                    raw = json.dumps(raw, ensure_ascii=False)
                except (TypeError, ValueError):
                    raw = str(raw)
            formatted_args = _format_tool_args(args) if name_counts[name] > 1 else ""
            segments.append(f"T:{name}({formatted_args})→{_format_tool_result(raw)}")

        body = " | ".join(segments)
        return f"[AT]\n{body}"

    @classmethod
    async def compress_messages(
        cls,
        old_messages: list[dict[str, Any]],
        *,
        model: str,
        api_key: str | None = None,
        completion_extra: dict[str, Any] | None = None,
    ) -> str:
        """Ask an LLM to rewrite ``old_messages`` as AAAK (single extra completion).

        When assistant messages already contain ``[AT]`` blocks (prior tool-round
        condensing), skips the LLM and emits deterministic ``USR:`` / passthrough
        ``[AT]`` lines so blocks are not double-encoded.
        """
        has_at_blocks = any(
            msg.get("role") == "assistant"
            and _message_content_to_text(msg).strip().startswith("[AT]")
            for msg in old_messages
        )
        if has_at_blocks:
            parts: list[str] = []
            for msg in old_messages:
                content = _message_content_to_text(msg).strip()
                role = msg.get("role", "unknown")
                if role == "assistant" and content.startswith("[AT]"):
                    parts.append(content)
                elif role == "user":
                    parts.append(f"USR: {content}")
                elif role == "assistant":
                    parts.append(f"AST: {content}")
                elif role == "tool":
                    tc_id = msg.get("tool_call_id", "")
                    parts.append(f"TOOL_RESULT[{tc_id}]: {content}")
                else:
                    parts.append(f"{role.upper()}: {content}")
            return "\n".join(parts)

        # Import when called so ``gluellm.api`` is fully initialized (avoids import cycles).
        from gluellm.api import _provider_cache

        transcript = transcript_from_messages(old_messages)
        compress_messages_api = [
            {"role": "system", "content": _COMPRESS_SYSTEM},
            {"role": "user", "content": _COMPRESS_USER_PREFIX + transcript},
        ]
        provider, model_id = _provider_cache.get_provider(model, api_key=api_key)
        extra = completion_extra or {}
        response = await provider.acompletion(
            model=model_id, messages=compress_messages_api, **extra
        )
        encoded = (response.choices[0].message.content or "").strip()
        if not encoded:
            logger.warning("AAAK compression returned empty content; caller should fall back.")
        return encoded
