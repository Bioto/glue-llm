"""Tests for JSON schema normalization for OpenAI compatibility.

These tests verify that Pydantic model schemas are correctly normalized
for OpenAI's structured output requirements.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict


class TestNormalizeSchemaForOpenAI:
    """Tests for the normalize_schema_for_openai function."""

    def test_basic_model(self):
        """Test normalization of a simple model."""
        from gluellm.schema import normalize_schema_for_openai

        class SimpleModel(BaseModel):
            name: str
            age: int

        schema = normalize_schema_for_openai(SimpleModel)

        # Check strict mode is enabled
        assert schema.get("strict") is True

        # Check additionalProperties is false
        assert schema.get("additionalProperties") is False

        # Check all fields are required
        assert set(schema.get("required", [])) == {"name", "age"}

    def test_nested_model(self):
        """Test normalization of nested models."""
        from gluellm.schema import normalize_schema_for_openai

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner
            name: str

        schema = normalize_schema_for_openai(Outer)

        # Check top-level
        assert schema.get("additionalProperties") is False
        assert schema.get("strict") is True

        # Check nested model in $defs
        defs = schema.get("$defs", {})
        if "Inner" in defs:
            assert defs["Inner"].get("additionalProperties") is False

    def test_union_type_in_list(self):
        """Test normalization of union types in lists - the original bug scenario."""
        from gluellm.schema import normalize_schema_for_openai

        class EntryA(BaseModel):
            type: str = "a"
            value_a: str

        class EntryB(BaseModel):
            type: str = "b"
            value_b: int

        class Container(BaseModel):
            items: list[EntryA | EntryB]

        schema = normalize_schema_for_openai(Container)

        # Should have $defs for both types
        defs = schema.get("$defs", {})
        assert "EntryA" in defs or "EntryB" in defs

        # Check additionalProperties is false on all nested types
        for name, defn in defs.items():
            assert defn.get("additionalProperties") is False, f"{name} should have additionalProperties: false"

    def test_model_with_extra_allow(self):
        """Test that extra='allow' (additionalProperties: true) is normalized to false."""
        from gluellm.schema import normalize_schema_for_openai

        class FlexibleModel(BaseModel):
            model_config = ConfigDict(extra="allow")
            name: str

        schema = normalize_schema_for_openai(FlexibleModel)

        # Should be normalized to false
        assert schema.get("additionalProperties") is False

    def test_optional_fields(self):
        """Test that optional fields are handled correctly."""
        from gluellm.schema import normalize_schema_for_openai

        class ModelWithOptional(BaseModel):
            required_field: str
            optional_field: str | None = None

        schema = normalize_schema_for_openai(ModelWithOptional)

        # Both fields should be in required (OpenAI strict mode requirement)
        required = set(schema.get("required", []))
        assert "required_field" in required
        assert "optional_field" in required

    def test_dict_with_any_type(self):
        """Test normalization of dict[str, Any] fields."""
        from gluellm.schema import normalize_schema_for_openai

        class ModelWithDict(BaseModel):
            metadata: dict[str, Any] | None = None

        schema = normalize_schema_for_openai(ModelWithDict)

        # Check the anyOf contains object type with additionalProperties: false
        properties = schema.get("properties", {})
        metadata_schema = properties.get("metadata", {})

        if "anyOf" in metadata_schema:
            for member in metadata_schema["anyOf"]:
                if member.get("type") == "object":
                    # Should have additionalProperties: false
                    assert member.get("additionalProperties") is False

    def test_no_booleans_except_strict_and_additional_properties(self):
        """Test that there are no unexpected boolean values in the schema."""
        from gluellm.schema import normalize_schema_for_openai

        class ComplexModel(BaseModel):
            name: str
            items: list[str]
            nested: dict[str, int] | None = None

        schema = normalize_schema_for_openai(ComplexModel)

        def check_booleans(obj, path=""):
            """Recursively check for booleans that might cause issues."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, bool):
                        # Only allowed booleans are:
                        # - strict: true
                        # - additionalProperties: false
                        if k == "strict":
                            assert v is True, f"strict should be True at {path}.{k}"
                        elif k == "additionalProperties":
                            assert v is False, f"additionalProperties should be False at {path}.{k}"
                        else:
                            pytest.fail(f"Unexpected boolean at {path}.{k}: {v}")
                    else:
                        check_booleans(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_booleans(item, f"{path}[{i}]")

        check_booleans(schema)


class TestCreateOpenAIResponseFormat:
    """Tests for the create_openai_response_format function."""

    def test_creates_proper_structure(self):
        """Test that the response format has the correct structure."""
        from gluellm.schema import create_openai_response_format

        class TestModel(BaseModel):
            value: int

        response_format = create_openai_response_format(TestModel)

        # Check top-level structure
        assert response_format["type"] == "json_schema"
        assert "json_schema" in response_format

        json_schema = response_format["json_schema"]
        assert json_schema["name"] == "TestModel"
        assert json_schema["strict"] is True
        assert "schema" in json_schema

    def test_schema_is_valid_json(self):
        """Test that the generated schema is valid JSON."""
        from gluellm.schema import create_openai_response_format

        class TestModel(BaseModel):
            name: str
            values: list[int]

        response_format = create_openai_response_format(TestModel)

        # Should be serializable to JSON without errors
        json_str = json.dumps(response_format)
        assert json_str is not None

        # Should be parseable back
        parsed = json.loads(json_str)
        assert parsed == response_format

    def test_non_strict_mode(self):
        """Test that strict mode can be disabled."""
        from gluellm.schema import create_openai_response_format

        class TestModel(BaseModel):
            value: int

        response_format = create_openai_response_format(TestModel, strict=False)

        assert response_format["json_schema"]["strict"] is False


class TestSchemaCompatibility:
    """Integration tests for schema compatibility with actual complex models."""

    def test_cash_flow_statement_scenario(self):
        """Test the exact scenario from the bug report."""
        from gluellm.schema import normalize_schema_for_openai

        class CashFlowStatementEntry(BaseModel):
            name: str
            amount: float

        class CashFlowStatementEntryGroup(BaseModel):
            name: str
            entries: list[CashFlowStatementEntry]

        class CashFlowStatement(BaseModel):
            items: list[CashFlowStatementEntry | CashFlowStatementEntryGroup]

        # This should not raise any errors
        schema = normalize_schema_for_openai(CashFlowStatement)

        # Verify structure
        assert "properties" in schema
        assert "items" in schema["properties"]
        assert schema["additionalProperties"] is False

        # Check all definitions have additionalProperties: false
        for _name, defn in schema.get("$defs", {}).items():
            if defn.get("type") == "object":
                assert defn.get("additionalProperties") is False

    def test_deeply_nested_unions(self):
        """Test deeply nested union types."""
        from gluellm.schema import normalize_schema_for_openai

        class Leaf(BaseModel):
            value: str

        class Branch(BaseModel):
            children: list["Leaf | Branch"]

        class Tree(BaseModel):
            root: Branch

        schema = normalize_schema_for_openai(Tree)

        # Should not crash and should have proper structure
        assert schema.get("strict") is True
        assert schema.get("additionalProperties") is False
