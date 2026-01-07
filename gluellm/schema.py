"""JSON Schema normalization for OpenAI structured outputs compatibility.

OpenAI's structured output feature has strict requirements for JSON schemas that
don't always align with what Pydantic generates. This module provides utilities
to normalize Pydantic model schemas for OpenAI compatibility.

Known issues addressed:
1. additionalProperties: true -> must be false for strict mode
2. Optional fields without defaults: must have explicit null handling
3. Missing 'required' arrays: all properties must be required in strict mode
4. $defs vs definitions: OpenAI uses definitions, not $defs
5. Unsupported keywords: title, description, examples need careful handling

Usage:
    from gluellm.schema import create_normalized_model, normalize_schema_for_openai

    class MyModel(BaseModel):
        name: str
        items: list[Item]

    # Recommended: Create a normalized model class (works with any_llm/OpenAI)
    NormalizedModel = create_normalized_model(MyModel)
    # Use NormalizedModel with structured_complete() - schema will be normalized

    # Alternative: Get normalized schema dict directly
    schema = normalize_schema_for_openai(MyModel)
"""

import copy
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def normalize_schema_for_openai(
    model: type[BaseModel],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Normalize a Pydantic model's JSON schema for OpenAI structured outputs.

    This function generates a JSON schema from a Pydantic model and transforms
    it to be compatible with OpenAI's structured output requirements.

    Args:
        model: The Pydantic model class to generate schema for
        strict: Whether to enforce OpenAI strict mode requirements (default: True)

    Returns:
        A normalized JSON schema dict compatible with OpenAI structured outputs

    Example:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     value: int
        ...     explanation: str
        ...
        >>> schema = normalize_schema_for_openai(Answer)
        >>> # Schema is now compatible with OpenAI's parse() method
    """
    # Generate the base schema from Pydantic
    schema = model.model_json_schema()

    # Deep copy to avoid modifying cached schemas
    schema = copy.deepcopy(schema)

    if strict:
        schema = _apply_strict_mode_fixes(schema)

    # Normalize $defs to definitions if needed (some OpenAI versions prefer this)
    schema = _normalize_defs(schema)

    # Add strict flag at root level for OpenAI
    if strict:
        schema["strict"] = True

    logger.debug(f"Normalized schema for {model.__name__}: keys={list(schema.keys())}")

    return schema


def _apply_strict_mode_fixes(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply fixes required for OpenAI strict mode.

    Strict mode requires:
    - additionalProperties: false on all objects
    - All properties must be in 'required' array
    - No unsupported schema features
    """

    def fix_object(obj: dict[str, Any], path: str = "") -> None:
        """Recursively fix an object schema."""
        if not isinstance(obj, dict):
            return

        obj_type = obj.get("type")

        # CRITICAL FIX: Remove 'required: True' from ANY schema (not just objects)
        # Pydantic can generate 'required: True' in field schemas which OpenAI rejects
        # OpenAI expects 'required' only at object level as an array, never as a boolean
        if "required" in obj and obj["required"] is True:
            # Remove invalid 'required: True' - it should only be an array at object level
            obj.pop("required")
            logger.debug(f"Removed invalid 'required: True' at {path}")

        # Fix object types
        if obj_type == "object":
            # Ensure additionalProperties is false, not true
            if obj.get("additionalProperties") is True:
                obj["additionalProperties"] = False
                logger.debug(f"Fixed additionalProperties at {path}")
            elif "additionalProperties" not in obj:
                obj["additionalProperties"] = False

            # CRITICAL: Remove 'required: True' from individual field schemas
            # Pydantic can generate field-level 'required: True' which OpenAI rejects
            # OpenAI expects 'required' only at the object level as an array
            properties = obj.get("properties", {})
            if properties:
                for prop_name, prop_schema in properties.items():
                    if isinstance(prop_schema, dict) and prop_schema.get("required") is True:
                        # Remove the invalid 'required: True' from field schema
                        prop_schema = copy.copy(prop_schema)
                        prop_schema.pop("required", None)
                        properties[prop_name] = prop_schema
                        logger.debug(f"Removed 'required: True' from field schema at {path}.properties.{prop_name}")

            # Ensure all properties are in required array (at object level, not field level)
            if properties:
                # Ensure 'required' is an array, not True
                current_required = obj.get("required")
                if current_required is True:
                    # Convert True to array of all properties
                    obj["required"] = list(properties.keys())
                    logger.debug(f"Converted 'required: True' to array at {path}")
                elif not isinstance(current_required, list):
                    # If it's not a list, make it one
                    obj["required"] = list(properties.keys())
                    logger.debug(f"Converted 'required' to array at {path}")

                required = set(obj.get("required", []))
                all_props = set(properties.keys())
                missing = all_props - required

                if missing:
                    # Add missing properties to required, handling optional fields
                    for prop_name in missing:
                        prop_schema = properties[prop_name]
                        # If not already nullable, make it nullable for optional fields
                        if not _is_nullable(prop_schema):
                            properties[prop_name] = _make_nullable(prop_schema)
                    obj["required"] = list(all_props)
                    logger.debug(f"Added {missing} to required at {path}")

        # Handle anyOf/oneOf containing null for optional types
        if "anyOf" in obj:
            obj["anyOf"] = [_fix_any_of_member(member, f"{path}.anyOf") for member in obj["anyOf"]]

        if "oneOf" in obj:
            obj["oneOf"] = [_fix_any_of_member(member, f"{path}.oneOf") for member in obj["oneOf"]]

        # Recurse into nested schemas
        for key in ["properties", "$defs", "definitions"]:
            if key in obj:
                for name, nested in obj[key].items():
                    fix_object(nested, f"{path}.{key}.{name}")

        # Handle array items
        if "items" in obj:
            items = obj["items"]
            if isinstance(items, dict):
                fix_object(items, f"{path}.items")
            elif isinstance(items, list):
                for i, item in enumerate(items):
                    fix_object(item, f"{path}.items[{i}]")

        # Handle allOf, anyOf, oneOf members
        for key in ["allOf", "anyOf", "oneOf"]:
            if key in obj:
                for i, member in enumerate(obj[key]):
                    fix_object(member, f"{path}.{key}[{i}]")

    fix_object(schema)
    return schema


def _fix_any_of_member(member: dict[str, Any], path: str) -> dict[str, Any]:
    """Fix an anyOf/oneOf member schema."""
    if isinstance(member, dict) and member.get("type") == "object" and member.get("additionalProperties") is True:
        member = copy.copy(member)
        member["additionalProperties"] = False
    return member


def _is_nullable(schema: dict[str, Any]) -> bool:
    """Check if a schema allows null values."""
    if schema.get("type") == "null":
        return True
    if "anyOf" in schema:
        return any(s.get("type") == "null" for s in schema["anyOf"] if isinstance(s, dict))
    if "oneOf" in schema:
        return any(s.get("type") == "null" for s in schema["oneOf"] if isinstance(s, dict))
    return False


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a schema nullable by wrapping in anyOf with null."""
    if _is_nullable(schema):
        return schema

    # If already an anyOf, add null type
    if "anyOf" in schema:
        schema = copy.copy(schema)
        schema["anyOf"] = list(schema["anyOf"]) + [{"type": "null"}]
        return schema

    # Wrap in anyOf
    return {
        "anyOf": [schema, {"type": "null"}],
        "default": None,
    }


def _normalize_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize $defs to work with OpenAI.

    OpenAI accepts $defs but some versions work better with references
    being consistent. This ensures internal consistency.
    """
    # OpenAI now supports $defs, so we just ensure consistency
    # If there are both $defs and definitions, merge them
    if "$defs" in schema and "definitions" in schema:
        schema["$defs"].update(schema.pop("definitions"))

    return schema


def create_normalized_model(
    model: type[BaseModel],
    *,
    strict: bool = True,
) -> type[BaseModel]:
    """Create a Pydantic model subclass with normalized JSON schema.

    Returns a subclass that overrides model_json_schema() to return
    an OpenAI-compatible schema, while preserving all other behavior.
    This is the recommended approach for fixing schema compatibility
    issues with OpenAI's structured outputs.

    Args:
        model: The Pydantic model class to normalize
        strict: Whether to enforce OpenAI strict mode requirements (default: True)

    Returns:
        A subclass of the original model with normalized schema generation

    Example:
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     items: list[str]
        ...
        >>> NormalizedMyModel = create_normalized_model(MyModel)
        >>> # Use NormalizedMyModel with OpenAI - schema will be normalized
        >>> # Response parsing still works because it's a subclass
    """
    # Generate normalized schema once
    normalized_schema = normalize_schema_for_openai(model, strict=strict)

    # Deep copy to ensure we don't modify the cached schema
    normalized_schema = copy.deepcopy(normalized_schema)

    # Create a subclass that overrides schema generation methods
    class NormalizedModel(model):
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Return the normalized schema instead of the original."""
            # Return a fresh copy to avoid any mutation issues
            return copy.deepcopy(normalized_schema)

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
            """Override Pydantic's internal schema generation method.

            This is called by Pydantic's schema generation system and ensures
            OpenAI SDK gets the normalized schema even if it uses this method.
            """
            # Return the normalized schema, converting it to the format expected
            # by the handler if needed
            # The handler expects a JsonSchemaValue, which is just a dict
            return copy.deepcopy(normalized_schema)

    # Preserve the original class name for OpenAI's schema naming
    NormalizedModel.__name__ = model.__name__
    NormalizedModel.__qualname__ = model.__qualname__
    NormalizedModel.__module__ = model.__module__

    logger.debug(f"Created normalized model class for {model.__name__}")

    return NormalizedModel


def create_openai_response_format(
    model: type[BaseModel],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Create the complete response_format dict for OpenAI API.

    This creates the full response_format structure expected by OpenAI's
    chat.completions.create() with structured outputs.

    Note: For use with any_llm and OpenAI's .parse() method, prefer
    create_normalized_model() instead, as it works better with the
    Pydantic model interface that OpenAI expects.

    Args:
        model: The Pydantic model class
        strict: Whether to use strict mode (default: True)

    Returns:
        A dict suitable for the response_format parameter

    Example:
        >>> response_format = create_openai_response_format(MyModel)
        >>> # Use directly with OpenAI client:
        >>> response = client.chat.completions.create(
        ...     messages=[...],
        ...     response_format=response_format,
        ... )
    """
    schema = normalize_schema_for_openai(model, strict=strict)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": schema,
            "strict": strict,
        },
    }
