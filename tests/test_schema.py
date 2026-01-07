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

    def test_removes_required_true_from_field_schemas(self):
        """Test that 'required: True' is removed from field schemas.

        Pydantic can generate 'required: True' in individual field schemas
        when using Annotated with Field(). OpenAI rejects this - 'required'
        must only be an array at the object level.
        """
        from typing import Annotated

        from pydantic import Field

        from gluellm.schema import normalize_schema_for_openai

        class TestModel(BaseModel):
            # Using Annotated with Field can cause 'required: True' in field schema
            name: Annotated[
                str,
                Field(
                    ...,
                    description="Name field",
                    examples=["test"],
                ),
            ]
            optional_field: Annotated[
                str | None,
                Field(
                    description="Optional field",
                    examples=[None],
                ),
            ] = None

        schema = normalize_schema_for_openai(TestModel)

        # Check that no field schema has 'required: True'
        def check_required_true(obj, path=""):
            """Recursively check for 'required: True'."""
            issues = []
            if isinstance(obj, dict):
                if "required" in obj and obj["required"] is True:
                    issues.append(f"{path}.required is True (should be array or removed)")
                for k, v in obj.items():
                    issues.extend(check_required_true(v, f"{path}.{k}"))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    issues.extend(check_required_true(item, f"{path}[{i}]"))
            return issues

        issues = check_required_true(schema)
        assert not issues, f"Found 'required: True' in schema: {issues}"

        # Verify top-level 'required' is an array
        assert isinstance(schema.get("required"), list), "Top-level 'required' should be a list"
        assert "name" in schema["required"]


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


class TestCreateNormalizedModel:
    """Tests for the create_normalized_model function."""

    def test_returns_subclass(self):
        """Test that the returned class is a subclass of the original."""
        from gluellm.schema import create_normalized_model

        class OriginalModel(BaseModel):
            name: str
            value: int

        normalized_model = create_normalized_model(OriginalModel)

        # Should be a subclass
        assert issubclass(normalized_model, OriginalModel)

        # Should be able to instantiate with same fields
        instance = normalized_model(name="test", value=42)
        assert instance.name == "test"
        assert instance.value == 42

    def test_overrides_schema_generation(self):
        """Test that model_json_schema() returns normalized schema."""
        from gluellm.schema import create_normalized_model, normalize_schema_for_openai

        class TestModel(BaseModel):
            name: str
            value: int

        normalized_model = create_normalized_model(TestModel)
        expected_schema = normalize_schema_for_openai(TestModel)

        # Should return normalized schema
        schema = normalized_model.model_json_schema()

        assert schema.get("strict") is True
        assert schema.get("additionalProperties") is False
        assert schema == expected_schema

    def test_preserves_class_name(self):
        """Test that the normalized class preserves the original name."""
        from gluellm.schema import create_normalized_model

        class IncomeStatement(BaseModel):
            revenue: float

        normalized_income_statement = create_normalized_model(IncomeStatement)

        # Should preserve name for OpenAI's schema naming
        assert normalized_income_statement.__name__ == "IncomeStatement"
        # __qualname__ may include function context, but __name__ is what OpenAI uses
        assert IncomeStatement.__name__ in normalized_income_statement.__qualname__

    def test_response_parsing_still_works(self):
        """Test that response parsing works with normalized model."""
        from gluellm.schema import create_normalized_model

        class TestModel(BaseModel):
            name: str
            count: int

        normalized_model = create_normalized_model(TestModel)

        # Parse from dict (simulating OpenAI response)
        data = {"name": "test", "count": 5}
        instance = normalized_model(**data)

        assert instance.name == "test"
        assert instance.count == 5
        assert isinstance(instance, TestModel)  # Should be instance of original too

    def test_works_with_complex_union_types(self):
        """Test that normalized model works with union types in lists."""
        from gluellm.schema import create_normalized_model

        class EntryA(BaseModel):
            type: str = "a"
            value: str

        class EntryB(BaseModel):
            type: str = "b"
            number: int

        class Container(BaseModel):
            items: list[EntryA | EntryB]

        normalized_container = create_normalized_model(Container)

        # Should generate normalized schema
        schema = normalized_container.model_json_schema()
        assert schema.get("strict") is True
        assert schema.get("additionalProperties") is False

        # Should still parse correctly
        data = {
            "items": [
                {"type": "a", "value": "test"},
                {"type": "b", "number": 42},
            ]
        }
        instance = normalized_container(**data)
        assert len(instance.items) == 2
        assert isinstance(instance.items[0], EntryA)
        assert isinstance(instance.items[1], EntryB)


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

    def test_cash_flow_statement_with_annotated_fields(self):
        """Test the exact user example with Annotated fields and Field().

        This reproduces the real-world scenario where Pydantic generates
        'required: True' in field schemas, which OpenAI rejects.
        """
        from typing import Annotated

        from pydantic import Field

        from gluellm.schema import create_normalized_model

        # Simulate the Date model
        class Date(BaseModel):
            year: int
            month: int
            day: int

            @classmethod
            def get_examples(cls, count: int = 1):
                return [Date(year=2025, month=9, day=30) for _ in range(count)]

        class CashFlowStatementEntry(BaseModel):
            name: str
            amount: float

        class CashFlowStatementEntryGroup(BaseModel):
            name: str
            entries: list[CashFlowStatementEntry]

        year = 2025

        class CashFlowStatement(BaseModel):
            # Using Date | None to match the pattern (the function would return Date type)
            reporting_date: Annotated[
                Date | None,
                Field(
                    description=(
                        "Date of reporting of the cash flow statement for the specific year, usually indicated by month"
                        "and day in the statement, and years as a column."
                    ),
                    examples=[
                        Date.get_examples(count=1)[0],
                        None,
                    ],
                ),
            ]
            scale: Annotated[
                int,
                Field(
                    description=(
                        "Scaling units of the values inside the single-year cash flow statement. Typically indicated by"
                        "'In millions', 'In billions', 'In thousands' etc. at the beginnig of the cash flow statement. For"
                        "instance, if the cash flow statement states (in millions), the scale is 1000000. If it states in "
                        "thousands, the scaling unit is 1000. If not scaling unit is indicated, this value should be 1. "
                        "If the scaling values differ for the years, find the one that covers the most entries sensibly. "
                        "For instance, if two years have millions, one has thousands, choose the  majority, millions."
                    ),
                    examples=[1000000, 1000000000, 1000, 1],
                ),
            ]
            unit: Annotated[
                str,
                Field(
                    ...,
                    description=(
                        "Unit of the majority of elements in the cash flow statement. This can be a specific currency,"
                        "physical unit or other. Percentage is not a unit, nor shares. Always use the ISO currency code for"
                        " currencies, such as USD, EUR, JPY, etc. If the units are very mixed with a lot of different "
                        "units, this is None. Usually, in cash flow statements, it's a currency like USD, EUR, JPY or "
                        "others."
                    ),
                    examples=["USD", "EUR", "JPY", "GBP", "CHF"],
                ),
            ]
            entries: Annotated[
                list[CashFlowStatementEntry | CashFlowStatementEntryGroup],
                Field(
                    description=(
                        "Entries of the cash flow statement for each year. This is a list of CashFlowStatementEntry or "
                        "CashFlowStatementEntryGroup."
                    ),
                ),
            ]

        # Create normalized model - this should not raise errors
        normalized_model = create_normalized_model(CashFlowStatement)

        # Get the schema
        schema = normalized_model.model_json_schema()

        # Verify no 'required: True' exists anywhere
        def check_required_true(obj, path=""):
            """Recursively check for 'required: True'."""
            issues = []
            if isinstance(obj, dict):
                if "required" in obj and obj["required"] is True:
                    issues.append(f"{path}.required is True (should be array or removed)")
                for k, v in obj.items():
                    issues.extend(check_required_true(v, f"{path}.{k}"))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    issues.extend(check_required_true(item, f"{path}[{i}]"))
            return issues

        issues = check_required_true(schema)
        assert not issues, f"Found 'required: True' in schema: {issues}"

        # Verify structure
        assert schema.get("additionalProperties") is False
        assert schema.get("strict") is True
        assert isinstance(schema.get("required"), list)
        assert "reporting_date" in schema["required"]
        assert "scale" in schema["required"]
        assert "unit" in schema["required"]
        assert "entries" in schema["required"]

        # Verify entries field has correct structure
        entries_schema = schema["properties"]["entries"]
        assert entries_schema["type"] == "array"
        assert "items" in entries_schema
        items_schema = entries_schema["items"]
        assert "anyOf" in items_schema  # Union type should be represented as anyOf

        # Verify it can still parse responses
        test_data = {
            "reporting_date": {"year": 2025, "month": 9, "day": 30},
            "scale": 1000000,
            "unit": "USD",
            "entries": [
                {"name": "Operating Activities", "amount": 1000000.0},
            ],
        }
        instance = normalized_model(**test_data)
        assert instance.scale == 1000000
        assert instance.unit == "USD"
        assert len(instance.entries) == 1

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
