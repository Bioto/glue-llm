"""Examples of tools with Pydantic model parameters.

gluellm automatically coerces LLM-generated JSON arguments to the Pydantic
model types declared in the tool's signature, so you can write tools that use
natural attribute access (``arg.field``) without any boilerplate.
"""

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from gluellm.api import complete

# ---------------------------------------------------------------------------
# Example 1 – flat Pydantic model parameter
# ---------------------------------------------------------------------------


class SearchConfig(BaseModel):
    """Configuration for a search operation."""

    query: Annotated[str, Field(description="The search query string")]
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 10


def search(config: SearchConfig) -> str:
    """Search for documents matching the given configuration.

    Args:
        config: Search configuration including query and max_results
    """
    # config is a SearchConfig instance — attribute access just works
    return f"Found {config.max_results} results for '{config.query}'"


async def example_flat_model():
    print("=" * 60)
    print("Example 1: Flat Pydantic model parameter")
    print("=" * 60)

    result = await complete(
        user_message="Search for 'async python' with a maximum of 5 results using the search tool.",
        system_prompt="You are a helpful assistant. Use the search tool when asked to search.",
        tools=[search],
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    for call in result.tool_execution_history:
        print(f"  search({call['arguments']}) -> {call['result']}")
    print()


# ---------------------------------------------------------------------------
# Example 2 – nested Pydantic models
# ---------------------------------------------------------------------------


class Address(BaseModel):
    """A postal address."""

    street: Annotated[str, Field(description="Street name and number")]
    city: Annotated[str, Field(description="City name")]
    postcode: Annotated[str, Field(description="Postal code")]


class Person(BaseModel):
    """A person with an address."""

    name: Annotated[str, Field(description="Full name")]
    age: Annotated[int, Field(description="Age in years")]
    address: Annotated[Address, Field(description="Home address")]


def register_person(person: Person) -> str:
    """Register a new person in the system.

    Args:
        person: The person to register, including their home address
    """
    # Both person and person.address are proper model instances
    return (
        f"Registered {person.name} (age {person.age}) "
        f"at {person.address.street}, {person.address.city} {person.address.postcode}"
    )


async def example_nested_models():
    print("=" * 60)
    print("Example 2: Nested Pydantic model parameters")
    print("=" * 60)

    result = await complete(
        user_message=(
            "Register Alice Smith, age 30, living at 42 Oak Street, "
            "Springfield, postcode SP1 2AB using the register_person tool."
        ),
        system_prompt="You are a helpful assistant. Use the register_person tool when asked to register people.",
        tools=[register_person],
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    for call in result.tool_execution_history:
        print(f"  register_person({call['arguments']}) -> {call['result']}")
    print()


# ---------------------------------------------------------------------------
# Example 3 – finance-style tool (the original motivating use case)
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A cited source for a data point."""

    id: Annotated[str, Field(description="Source document identifier")]
    quote: Annotated[str, Field(description="Relevant excerpt from the source")]


class Quantity(BaseModel):
    """A financial quantity with unit and supporting sources."""

    value: Annotated[float, Field(description="Numeric value")]
    unit: Annotated[str, Field(description="Unit, e.g. 'USD million' or 'EUR billion'")]
    sources: Annotated[list[Source], Field(description="Supporting citations")] = []


def report_quantity(qty: Quantity) -> str:
    """Report a financial quantity with its sources.

    Args:
        qty: The quantity to report, including value, unit and source citations
    """
    source_ids = [s.id for s in qty.sources]
    sources_str = f" (sources: {', '.join(source_ids)})" if source_ids else ""
    return f"Reported: {qty.value:,.0f} {qty.unit}{sources_str}"


async def example_finance_tool():
    print("=" * 60)
    print("Example 3: Finance-style tool with nested sources")
    print("=" * 60)

    result = await complete(
        user_message=(
            "Report a quantity of 5559 USD million using the report_quantity tool. "
            "Cite source id 'annual-report-2024' with quote 'Revenue was $5,559M'."
        ),
        system_prompt="You are a financial data assistant. Use the report_quantity tool to log figures.",
        tools=[report_quantity],
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    for call in result.tool_execution_history:
        print(f"  report_quantity({call['arguments']}) -> {call['result']}")
    print()


# ---------------------------------------------------------------------------
# Example 4 – mixed parameters (Pydantic + primitives)
# ---------------------------------------------------------------------------


class FilterOptions(BaseModel):
    """Options for filtering a dataset."""

    min_score: Annotated[float, Field(description="Minimum score threshold")] = 0.0
    tags: Annotated[list[str], Field(description="Tags to filter by")] = []
    active_only: Annotated[bool, Field(description="Only include active records")] = True


def query_dataset(dataset_name: str, limit: int, filters: FilterOptions) -> str:
    """Query a named dataset with optional filters.

    Args:
        dataset_name: Name of the dataset to query
        limit: Maximum number of records to return
        filters: Filtering options to apply
    """
    # Primitive params (str, int) arrive as-is; FilterOptions is auto-coerced
    tag_str = f", tags={filters.tags}" if filters.tags else ""
    active_str = " (active only)" if filters.active_only else ""
    return (
        f"Queried '{dataset_name}'{active_str}: up to {limit} records "
        f"with min_score≥{filters.min_score}{tag_str}"
    )


async def example_mixed_params():
    print("=" * 60)
    print("Example 4: Mixed Pydantic + primitive parameters")
    print("=" * 60)

    result = await complete(
        user_message=(
            "Query the 'products' dataset, limit 20 records, "
            "min score 0.8, tags ['electronics', 'sale'], active records only, "
            "using the query_dataset tool."
        ),
        system_prompt="You are a data assistant. Use the query_dataset tool when asked to query data.",
        tools=[query_dataset],
    )

    print(f"Response: {result.final_response}")
    print(f"Tool calls made: {result.tool_calls_made}")
    for call in result.tool_execution_history:
        print(f"  query_dataset({call['arguments']}) -> {call['result']}")
    print()


if __name__ == "__main__":
    async def main():
        await example_flat_model()
        await example_nested_models()
        await example_finance_tool()
        await example_mixed_params()

    asyncio.run(main())
