"""
Example: Multimodal image input via the OpenAI Responses API.

The Responses API accepts image input in three flavours, all sent inside an
``input_image`` content part of a user message:

1. **Public URL** — the simplest case. Pass an HTTPS URL.
2. **Base64 data URL** — fully self-contained. Read the file, base64-encode it,
   and prefix with ``data:<mime>;base64,``. Useful for local files when you
   don't want to upload to OpenAI first.
3. **OpenAI file_id** — upload the file once via the Files API
   (``purpose="vision"``) and reference it by id on subsequent requests. Best
   for images you'll reuse across many turns or sessions.

You can also tune cost vs fidelity with the ``detail`` field on each
``input_image`` part: ``"low"`` (fewer tokens, faster), ``"high"`` (more
tokens, sharper analysis), or ``"auto"`` (default — GlueLLM fills this when
omitted).

Both :func:`gluellm.response` (plain text) and
:func:`gluellm.structured_response` (Pydantic) accept multimodal input —
the helpers round-trip the typed content list through the wire layer.
"""

import asyncio
import base64
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gluellm import APIConnectionError, APITimeoutError, InvalidRequestError, response, structured_response

# A stable Wikimedia image used in OpenAI's own documentation. Boardwalk
# scene through Wisconsin wetlands — easy for a vision model to describe.
BOARDWALK_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)

# A tiny 1x1 red-pixel PNG, hardcoded so the base64 example is fully
# self-contained and can run offline-of-storage. Replace with your own image
# in real usage via ``encode_local_image()``.
_TINY_RED_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
_TINY_RED_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(_TINY_RED_PNG_BYTES).decode("ascii")
_OPTIONAL_PUBLIC_URL_ERRORS = (InvalidRequestError, APIConnectionError, APITimeoutError)


def encode_local_image(path: str | Path) -> str:
    """Encode a local image file as a base64 ``data:`` URL.

    This is the pattern you'll use for your own files. The Responses API
    accepts both ``https://...`` URLs and ``data:<mime>;base64,...`` URLs in
    the ``image_url`` field of an ``input_image`` content part.

    Args:
        path: Path to a local image file. Extension determines MIME type.

    Returns:
        A ``data:<mime>;base64,<payload>`` string ready to drop into an
        ``input_image`` part.
    """
    p = Path(path)
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_by_suffix.get(p.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image extension: {p.suffix!r}")
    payload = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


async def run_optional_public_url_example(name: str, example):
    """Run a public-URL demo without making the full example suite flaky."""
    try:
        await example()
    except _OPTIONAL_PUBLIC_URL_ERRORS as e:
        print(f"({name} skipped: {type(e).__name__}: {e})\n")


# Example 1: Image via public URL
async def example_image_from_url():
    """The simplest pattern: drop an HTTPS image URL into an input_image part."""
    print("=" * 80)
    print("Example 1: Image from a Public URL")
    print("=" * 80)

    result = await response(
        user_input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this scene in one sentence."},
                    {"type": "input_image", "image_url": BOARDWALK_URL},
                ],
            }
        ],
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}")
    print(f"Tokens used: {result.tokens_used}")
    print()


# Example 2: Local image as a base64 data URL
async def example_image_from_base64():
    """Encode a local file inline. Best when you can't expose a public URL.

    The example writes a tiny placeholder PNG to a temp file so it runs
    self-contained. In real usage replace ``tmp_image_path`` with your own
    image path and call ``encode_local_image(...)`` on it.
    """
    print("=" * 80)
    print("Example 2: Image from a Local File (Base64 Data URL)")
    print("=" * 80)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_RED_PNG_BYTES)
        tmp_image_path = Path(f.name)

    try:
        data_url = encode_local_image(tmp_image_path)
        print(f"(encoded {tmp_image_path.name} as data URL of length {len(data_url)})")

        result = await response(
            user_input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "What colour is this tiny image? Answer in one word.",
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            model="openai:gpt-5.4-2026-03-05",
        )

        print(f"Response: {result.final_response}")
        print(f"Tokens used: {result.tokens_used}")
        print()
    finally:
        tmp_image_path.unlink(missing_ok=True)


# Example 3: Cost optimization with the `detail` parameter
async def example_image_with_detail_low():
    """Set ``detail="low"`` to reduce vision tokens for fast / cheap analysis.

    The Responses API tokenises images by tile. ``detail="low"`` uses a
    single low-res tile (~85 tokens), ``detail="high"`` uses many tiles
    (often 1000+ tokens), and ``detail="auto"`` (default) chooses based on
    image size.
    """
    print("=" * 80)
    print("Example 3: Cost-Optimised Image Input (detail='low')")
    print("=" * 80)

    result = await response(
        user_input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Is this an indoor or outdoor scene? One word."},
                    {
                        "type": "input_image",
                        "image_url": _TINY_RED_PNG_DATA_URL,
                        "detail": "low",
                    },
                ],
            }
        ],
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}")
    print(f"Tokens used: {result.tokens_used}")
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")
    print()


# Example 4: file_id pattern via the OpenAI Files API
async def example_image_from_file_id():
    """Upload once via the Files API, reference by ``file_id`` thereafter.

    Best when you'll re-use the same image across many requests/sessions:
    you pay the upload cost once and avoid re-sending bytes on every call.
    Requires ``purpose="vision"`` for image inputs.
    """
    print("=" * 80)
    print("Example 4: Image via OpenAI file_id (Files API upload)")
    print("=" * 80)

    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_RED_PNG_BYTES)
        tmp_image_path = Path(f.name)

    file_id: str | None = None
    try:
        with tmp_image_path.open("rb") as fh:
            uploaded = await client.files.create(file=fh, purpose="vision")
        file_id = uploaded.id
        print(f"Uploaded file_id: {file_id}")

        result = await response(
            user_input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What colour is this tiny image?"},
                        {"type": "input_image", "file_id": file_id},
                    ],
                }
            ],
            model="openai:gpt-5.4-2026-03-05",
        )

        print(f"Response: {result.final_response}")
        print(f"Tokens used: {result.tokens_used}")
        print()
    finally:
        tmp_image_path.unlink(missing_ok=True)
        if file_id is not None:
            try:
                await client.files.delete(file_id)
            except Exception as cleanup_error:
                print(f"(cleanup warning: failed to delete {file_id}: {cleanup_error})")
        await client.close()


# Example 5: Structured output from an image
class ImageDescription(BaseModel):
    """Typed analysis of a single image."""

    summary: Annotated[str, Field(description="One-sentence summary of the image")]
    primary_subject: Annotated[str, Field(description="The main subject of the image")]
    setting: Annotated[
        str, Field(description="Where this scene takes place (e.g. 'outdoor wetland', 'indoor office')")
    ]
    dominant_colors: Annotated[
        list[str], Field(description="2-4 dominant colours in the image", min_length=1, max_length=4)
    ]


async def example_structured_response_with_image():
    """Combine multimodal input with Pydantic-validated output.

    ``structured_response()`` works identically to ``response()`` for input —
    just pass a typed user message with an ``input_image`` part. The model's
    final answer is then parsed into your Pydantic schema, with all the usual
    validation retries and tool support available.
    """
    print("=" * 80)
    print("Example 5: Structured Response from an Image")
    print("=" * 80)

    result = await structured_response(
        user_input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyse this image and fill in the schema.",
                    },
                    {"type": "input_image", "image_url": _TINY_RED_PNG_DATA_URL},
                ],
            }
        ],
        response_format=ImageDescription,
        system_prompt="You are an art critic. Be concise but precise.",
        model="openai:gpt-5.4-2026-03-05",
    )

    if result.structured_output is None:
        raise RuntimeError("Model did not return structured output")

    desc = result.structured_output
    print(f"Summary:         {desc.summary}")
    print(f"Primary subject: {desc.primary_subject}")
    print(f"Setting:         {desc.setting}")
    print(f"Dominant colors: {', '.join(desc.dominant_colors)}")
    print(f"\nTokens used: {result.tokens_used}")
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")
    print()


# Example 6: Multiple images in a single request (comparison)
async def example_compare_two_images():
    """Send multiple ``input_image`` parts in the same user message.

    The Responses API accepts an arbitrary number of image parts per
    message; the model treats them as a sequence. Perfect for comparison
    tasks: "which of these is X?", "spot the difference", before/after,
    etc.
    """
    print("=" * 80)
    print("Example 6: Comparing Multiple Images in One Request")
    print("=" * 80)

    # Same image twice for the demo; in real use these would differ.
    result = await response(
        user_input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "I'm showing you two images. Compare them: are they "
                            "the same, similar, or different? Explain in one sentence."
                        ),
                    },
                    {"type": "input_image", "image_url": _TINY_RED_PNG_DATA_URL},
                    {"type": "input_image", "image_url": _TINY_RED_PNG_DATA_URL},
                ],
            }
        ],
        model="openai:gpt-5.4-2026-03-05",
    )

    print(f"Response: {result.final_response}")
    print(f"Tokens used: {result.tokens_used}")
    print()


async def main():
    """Run all examples."""
    print("\nMultimodal Image Examples (OpenAI Responses API)\n")

    await run_optional_public_url_example("Example 1", example_image_from_url)
    await example_image_from_base64()
    await example_image_with_detail_low()

    # The Files-API upload requires extra permissions; skip gracefully if it
    # isn't available on this account so the example file as a whole still
    # passes the integration test runner.
    try:
        await example_image_from_file_id()
    except Exception as e:
        print(f"(Example 4 skipped: {type(e).__name__}: {e})\n")

    await example_structured_response_with_image()
    await example_compare_two_images()

    print("=" * 80)
    print("All examples completed.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
