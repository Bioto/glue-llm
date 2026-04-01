"""Tests for tool behavior when parameters are typed as Pydantic models.

gluellm / any_llm automatically coerces tool arguments annotated as
``BaseModel`` subclasses to the correct model instance before invoking the
tool.  Tools can use natural attribute access everywhere — no boilerplate
needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from gluellm.api import ExecutionResult, complete

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Shared models used across tests
# ---------------------------------------------------------------------------


class SearchConfig(BaseModel):
    """Configuration for a search operation."""

    query: Annotated[str, Field(description="The search query string")]
    max_results: Annotated[int, Field(description="Maximum number of results to return")] = 10


class Source(BaseModel):
    """A citation source."""

    id: Annotated[str, Field(description="Source identifier")]
    quote: Annotated[str, Field(description="Relevant quote from the source")]


class Quantity(BaseModel):
    """A financial or scientific quantity with optional citations."""

    value: Annotated[float, Field(description="Numeric value")]
    unit: Annotated[str, Field(description="Unit of measurement, e.g. USD million")]
    sources: Annotated[list[Source], Field(description="Citations for this value")] = []


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


# ---------------------------------------------------------------------------
# Test 1 – basic Pydantic model parameter: what type does the tool receive?
# ---------------------------------------------------------------------------


class TestPydanticModelParameterType:
    """Verify the runtime type a tool receives for a Pydantic-annotated parameter."""

    async def test_tool_receives_model_instance_not_dict(self):
        """A tool typed with a Pydantic model receives a model instance at runtime.

        gluellm inspects the function's type annotations after ``json.loads``
        and coerces dict arguments to the declared ``BaseModel`` subclass via
        ``model_validate`` before invoking the tool.
        """
        received: list[object] = []

        def run_search(config: SearchConfig) -> str:
            """Run a search with the given configuration.

            Args:
                config: Search configuration including query and max_results
            """
            received.append(config)
            return f"received type={type(config).__name__}"

        result = await complete(
            user_message=(
                "Run a search for 'python tutorials' with max 5 results "
                "using the run_search tool."
            ),
            system_prompt=(
                "You are a helpful assistant. "
                "You MUST use the run_search tool to answer any search request."
            ),
            tools=[run_search],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1, "Expected at least one tool call"

        assert len(received) >= 1, "Tool should have been called at least once"
        actual = received[0]

        assert isinstance(actual, SearchConfig), (
            f"Expected a SearchConfig instance but got {type(actual).__name__}. "
            "gluellm should coerce JSON dicts to the annotated Pydantic model."
        )

    async def test_accessing_model_attribute_works_without_workaround(self):
        """Natural attribute access on a Pydantic-annotated parameter works.

        With automatic coercion, tools can use ``config.query`` directly
        without defensive dict-key access or manual ``model_validate`` calls.
        """
        results: list[str] = []

        def search_naive(config: SearchConfig) -> str:
            """Search with the given configuration.

            Args:
                config: Search configuration including query and max_results
            """
            # Natural attribute access — no workaround needed
            results.append(config.query)
            return f"Searching for: {config.query}"

        result = await complete(
            user_message="Search for 'async python' using the search_naive tool",
            system_prompt="You are a helpful assistant. Use the search_naive tool when asked to search.",
            tools=[search_naive],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(results) >= 1, "Tool should have been called"
        assert isinstance(results[0], str)
        assert len(results[0]) > 0



# ---------------------------------------------------------------------------
# Test 2 – nested Pydantic model as tool parameter
# ---------------------------------------------------------------------------


class TestNestedPydanticModelParameter:
    """Verify behaviour when a tool parameter is a model with nested models."""

    async def test_nested_model_arrives_as_model_instance(self):
        """A tool with a nested Pydantic model parameter receives a proper model instance.

        ``Person.model_validate`` is called automatically, which also coerces
        the nested ``address`` dict into an ``Address`` instance — so
        ``person.address.city`` works without any manual coercion.
        """
        received_persons: list[object] = []

        def register_person(person: Person) -> str:
            """Register a person in the system.

            Args:
                person: The person to register including their address
            """
            received_persons.append(person)
            return f"Registered {person.name} in {person.address.city}"

        result = await complete(
            user_message=(
                "Register Alice Smith, age 30, at 42 Oak Street, Springfield, postcode SP1 2AB "
                "using the register_person tool"
            ),
            system_prompt="You are a helpful assistant. Use the register_person tool when asked to register people.",
            tools=[register_person],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received_persons) >= 1

        received = received_persons[0]
        assert isinstance(received, Person), (
            f"Expected a Person instance, got {type(received).__name__}"
        )
        # Nested model is also coerced — not left as a plain dict
        assert isinstance(received.address, Address), (
            f"Expected nested address to be an Address instance, got {type(received.address).__name__}"
        )

    async def test_nested_model_attribute_access_works_at_depth(self):
        """Attribute access on the nested model works after automatic coercion.

        ``person.address.city`` succeeds because both the outer and nested
        models are coerced to proper instances before the tool is called.
        """
        cities: list[str] = []

        def describe_person(person: Person) -> str:
            """Describe a person and their location.

            Args:
                person: The person to describe
            """
            city = person.address.city
            cities.append(city)
            return f"Lives in {city}"

        result = await complete(
            user_message="Describe Bob Jones, age 45, at 7 Pine Ave, London, postcode EC1A 1BB using describe_person",
            system_prompt="You are a helpful assistant. Use the describe_person tool when asked to describe people.",
            tools=[describe_person],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(cities) >= 1, "Tool should have been called"
        assert isinstance(cities[0], str)
        assert len(cities[0]) > 0


# ---------------------------------------------------------------------------
# Test 3 – scalar/primitive parameters work as expected (regression guard)
# ---------------------------------------------------------------------------


class TestScalarParameterBaseline:
    """Confirm that scalar-typed tool parameters are unaffected by this issue.

    Primitive types (str, int, bool, list) are passed through directly from
    JSON and work as expected — this section acts as a sanity baseline.
    """

    async def test_str_parameter_works(self):
        """str parameters come through correctly."""
        received: list[str] = []

        def echo(message: str) -> str:
            """Echo the message back.

            Args:
                message: The message to echo
            """
            received.append(message)
            return f"Echo: {message}"

        result = await complete(
            user_message="Echo the message 'hello world' using the echo tool",
            system_prompt="You are a helpful assistant. Use the echo tool when asked to echo.",
            tools=[echo],
        )

        assert result.tool_calls_made >= 1
        assert len(received) >= 1
        assert isinstance(received[0], str)

    async def test_int_parameter_works(self):
        """int parameters come through correctly."""
        received: list[int] = []

        def double(n: int) -> int:
            """Double a number.

            Args:
                n: The number to double
            """
            received.append(n)
            return n * 2

        result = await complete(
            user_message="Double the number 21 using the double tool",
            system_prompt="You are a helpful assistant. Use the double tool when asked to double a number.",
            tools=[double],
        )

        assert result.tool_calls_made >= 1
        assert len(received) >= 1
        assert isinstance(received[0], int)
        assert received[0] == 21

    async def test_mixed_scalar_parameters_work(self):
        """Multiple primitive parameters come through with correct types."""
        received: list[tuple] = []

        def greet(name: str, times: int, loud: bool = False) -> str:
            """Greet someone.

            Args:
                name: Name of the person to greet
                times: How many times to greet them
                loud: Whether to use uppercase
            """
            received.append((name, times, loud))
            greeting = f"Hello {name}! " * times
            return greeting.upper() if loud else greeting.strip()

        result = await complete(
            user_message="Greet 'Alice' 2 times loudly using the greet tool",
            system_prompt="You are a helpful assistant. Use the greet tool when asked to greet someone.",
            tools=[greet],
        )

        assert result.tool_calls_made >= 1
        assert len(received) >= 1
        name, times, loud = received[0]
        assert isinstance(name, str)
        assert isinstance(times, int)
        assert isinstance(loud, bool)


# ---------------------------------------------------------------------------
# Additional models for complex-scenario tests
# ---------------------------------------------------------------------------


class Priority(str, Enum):
    """Task priority level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Tag(BaseModel):
    """A label attached to a task."""

    name: Annotated[str, Field(description="Tag label")]
    color: Annotated[str, Field(description="Hex color code, e.g. #ff0000")] = "#888888"


class Task(BaseModel):
    """A project task."""

    title: Annotated[str, Field(description="Short title of the task")]
    description: Annotated[str, Field(description="Detailed description")] = ""
    priority: Annotated[Priority, Field(description="Priority: low, medium, high, or critical")]
    tags: Annotated[list[Tag], Field(description="List of tags attached to this task")] = []
    assignee: Annotated[str | None, Field(description="Username of the assignee, or null")] = None


class MoneyAmount(BaseModel):
    """A monetary amount with currency."""

    amount: Annotated[float, Field(description="Numeric amount", gt=0)]
    currency: Annotated[str, Field(description="ISO 4217 currency code, e.g. USD", min_length=3, max_length=3)]

    @field_validator("currency")
    @classmethod
    def currency_must_be_uppercase(cls, v: str) -> str:
        return v.upper()


class LineItem(BaseModel):
    """A single line item on an invoice."""

    product: Annotated[str, Field(description="Product name")]
    quantity: Annotated[int, Field(description="Number of units", ge=1)]
    unit_price: Annotated[MoneyAmount, Field(description="Price per unit")]


class Invoice(BaseModel):
    """A customer invoice."""

    invoice_number: Annotated[str, Field(description="Unique invoice identifier")]
    customer_name: Annotated[str, Field(description="Full name of the customer")]
    items: Annotated[list[LineItem], Field(description="Line items on this invoice", min_length=1)]
    discount_percent: Annotated[float, Field(description="Discount percentage 0-100", ge=0, le=100)] = 0.0

    @model_validator(mode="after")
    def at_least_one_item(self) -> Invoice:
        if not self.items:
            raise ValueError("Invoice must have at least one line item")
        return self


class EmailNotification(BaseModel):
    """Email notification request."""

    to: Annotated[list[str], Field(description="List of recipient email addresses")]
    subject: Annotated[str, Field(description="Email subject line")]
    body: Annotated[str, Field(description="Email body text")]
    cc: Annotated[list[str], Field(description="CC recipients")] = []
    priority: Annotated[Literal["normal", "urgent"], Field(description="Delivery priority: normal or urgent")] = "normal"


# ---------------------------------------------------------------------------
# Test 5 – model with an Enum field
# ---------------------------------------------------------------------------


class TestEnumFieldInModel:
    """Models with Enum fields: the LLM sends a string value, Pydantic coerces it."""

    async def test_enum_field_auto_coerced_to_priority_member(self):
        """An Enum-typed field is coerced to the proper enum member automatically.

        any_llm inspects the function annotation and coerces the JSON string
        value to the declared ``Priority`` enum member before invoking the tool.
        """
        received: list[object] = []

        def create_task(task: Task) -> str:
            """Create a new project task.

            Args:
                task: The task to create with title, priority, and optional tags
            """
            received.append(task)
            return f"Created task '{task.title}'"

        result = await complete(
            user_message="Create a high-priority task titled 'Fix login bug' using the create_task tool",
            system_prompt="You are a helpful assistant. Use the create_task tool when asked to create tasks.",
            tools=[create_task],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        task = received[0]
        assert isinstance(task, Task), (
            f"Expected Task instance, got {type(task).__name__}"
        )
        # Enum field is already a proper Priority member, not a raw string
        assert isinstance(task.priority, Priority)
        assert task.priority in Priority



# ---------------------------------------------------------------------------
# Test 6 – model with a list of nested models
# ---------------------------------------------------------------------------


class TestListOfNestedModels:
    """Models containing a list of nested Pydantic models."""

    async def test_list_of_nested_models_auto_coerced_to_task_with_tag_instances(self):
        """A list[NestedModel] field is coerced to a list of proper model instances.

        any_llm coerces the entire Task, including its ``list[Tag]`` field, so
        every tag arrives as a ``Tag`` instance with attribute access working.
        """
        received: list[object] = []

        def create_task_with_tags(task: Task) -> str:
            """Create a task that may have several tags.

            Args:
                task: The task to create, which may include a list of tags
            """
            received.append(task)
            tag_summary = ", ".join(t.name for t in task.tags)
            return f"Task '{task.title}' with tags: {tag_summary}"

        result = await complete(
            user_message=(
                "Create a medium-priority task 'Update docs' with tags 'docs' (blue #0000ff) "
                "and 'maintenance' (green #00ff00) using create_task_with_tags"
            ),
            system_prompt="You are a helpful assistant. Use the create_task_with_tags tool when asked.",
            tools=[create_task_with_tags],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        task = received[0]
        assert isinstance(task, Task), f"Expected Task, got {type(task).__name__}"
        assert isinstance(task.tags, list)
        for tag in task.tags:
            assert isinstance(tag, Tag), (
                f"Expected Tag instance in tags list, got {type(tag).__name__}"
            )



# ---------------------------------------------------------------------------
# Test 7 – deeply nested model with field validators
# ---------------------------------------------------------------------------


class TestDeeplyNestedModelWithValidators:
    """Three-level nesting: Invoice → LineItem → MoneyAmount, plus validators."""

    async def test_deeply_nested_model_auto_coerced_at_all_levels(self):
        """Invoice, LineItem, and MoneyAmount are all auto-coerced to model instances.

        any_llm recursively coerces the entire three-level graph so the tool
        can use attribute access (``invoice.items[0].unit_price.amount``) directly.
        """
        received: list[object] = []

        def process_invoice(invoice: Invoice) -> str:
            """Process a customer invoice.

            Args:
                invoice: The invoice to process with customer info and line items
            """
            received.append(invoice)
            return f"Invoice {invoice.invoice_number} with {len(invoice.items)} item(s)"

        result = await complete(
            user_message=(
                "Process invoice INV-001 for customer 'Acme Corp' with one item: "
                "5 units of 'Widget' at USD 9.99 each. Use the process_invoice tool."
            ),
            system_prompt="You are a helpful assistant. Use the process_invoice tool when asked to process invoices.",
            tools=[process_invoice],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        invoice = received[0]
        assert isinstance(invoice, Invoice), f"Expected Invoice, got {type(invoice).__name__}"
        assert len(invoice.items) >= 1

        first_item = invoice.items[0]
        assert isinstance(first_item, LineItem), (
            f"Expected LineItem, got {type(first_item).__name__}"
        )
        assert isinstance(first_item.unit_price, MoneyAmount), (
            f"Expected MoneyAmount, got {type(first_item.unit_price).__name__}"
        )

    async def test_field_and_model_validators_fire_on_coercion(self):
        """Pydantic validators run automatically as part of the coercion step.

        The ``@field_validator`` on ``MoneyAmount`` uppercases the currency, and
        the ``@model_validator`` on ``Invoice`` enforces at least one item — both
        fire without any explicit ``model_validate`` call in the tool.
        """
        received: list[Invoice] = []

        def process_invoice(invoice: Invoice) -> str:
            """Process a customer invoice.

            Args:
                invoice: The invoice to process with customer info and line items
            """
            received.append(invoice)
            total = sum(item.quantity * item.unit_price.amount for item in invoice.items)
            currency = invoice.items[0].unit_price.currency
            return f"Invoice {invoice.invoice_number}: total={total:.2f} {currency}"

        result = await complete(
            user_message=(
                "Process invoice INV-042 for 'Beta Ltd' with two items: "
                "3 'Gadgets' at usd 14.50 each and 1 'Service' at eur 99.00. "
                "Use the process_invoice tool."
            ),
            system_prompt="You are a helpful assistant. Use the process_invoice tool when asked.",
            tools=[process_invoice],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        invoice = received[0]
        assert isinstance(invoice, Invoice)
        for item in invoice.items:
            assert isinstance(item, LineItem)
            assert isinstance(item.unit_price, MoneyAmount)
            assert item.unit_price.currency == item.unit_price.currency.upper(), (
                "@field_validator did not uppercase the currency"
            )


# ---------------------------------------------------------------------------
# Test 8 – model with optional fields and Literal type
# ---------------------------------------------------------------------------


class TestOptionalFieldsAndLiterals:
    """Models with Optional fields and Literal type constraints."""

    async def test_optional_field_defaults_applied_on_coerced_model(self):
        """Optional fields with defaults are applied during auto-coercion.

        When the LLM omits an optional field, the coerced model instance has
        the Pydantic default value rather than requiring manual fallback logic.
        """
        received: list[object] = []

        def send_email(notification: EmailNotification) -> str:
            """Send an email notification.

            Args:
                notification: The email to send with recipients, subject, and body
            """
            received.append(notification)
            return f"Sent to {notification.to} [cc={notification.cc}, priority={notification.priority}]"

        result = await complete(
            user_message=(
                "Send an email to ['alice@example.com'] with subject 'Hello' "
                "and body 'Hi there' using the send_email tool. Don't include CC."
            ),
            system_prompt="You are a helpful assistant. Use the send_email tool when asked to send emails.",
            tools=[send_email],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        notif = received[0]
        assert isinstance(notif, EmailNotification), (
            f"Expected EmailNotification, got {type(notif).__name__}"
        )
        # 'to' should be a list of strings
        assert isinstance(notif.to, list)
        assert all(isinstance(addr, str) for addr in notif.to)
        # cc defaults to [] when omitted
        assert isinstance(notif.cc, list)

    async def test_literal_field_constrained_by_coercion(self):
        """A Literal-typed field is validated to one of the allowed values on coercion."""
        received: list[EmailNotification] = []

        def send_email_urgent(notification: EmailNotification) -> str:
            """Send an email notification.

            Args:
                notification: The email to send, fully validated
            """
            received.append(notification)
            return f"Sent urgent={notification.priority == 'urgent'} to {notification.to}"

        result = await complete(
            user_message=(
                "Send an urgent email to ['bob@example.com'] with subject 'Alert' "
                "and body 'System down' using send_email_urgent."
            ),
            system_prompt="You are a helpful assistant. Use the send_email_urgent tool when asked to send emails.",
            tools=[send_email_urgent],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        notif = received[0]
        assert isinstance(notif, EmailNotification)
        assert notif.priority in ("normal", "urgent")
        assert isinstance(notif.to, list)

    async def test_task_with_none_assignee_coerced_to_model(self):
        """An Optional[str] field omitted by the LLM is None on the coerced model."""
        received: list[object] = []

        def assign_task(task: Task) -> str:
            """Assign a task to someone (or leave unassigned).

            Args:
                task: The task, which may have a null assignee if unassigned
            """
            received.append(task)
            return f"Task '{task.title}' assigned to {task.assignee!r}"

        result = await complete(
            user_message=(
                "Create a medium-priority task 'Write tests' with no assignee "
                "using the assign_task tool."
            ),
            system_prompt="You are a helpful assistant. Use the assign_task tool when asked.",
            tools=[assign_task],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        task = received[0]
        assert isinstance(task, Task), f"Expected Task, got {type(task).__name__}"
        # assignee should be None (Pydantic default) or a string if LLM provided one
        assert task.assignee is None or isinstance(task.assignee, str)


# ---------------------------------------------------------------------------
# Complex models for deep multi-level relationship tests
# ---------------------------------------------------------------------------


class ContactInfo(BaseModel):
    """Contact details for a person or organisation."""

    email: Annotated[str, Field(description="Email address")]
    phone: Annotated[str | None, Field(description="Phone number, or null")] = None
    linkedin: Annotated[str | None, Field(description="LinkedIn profile URL, or null")] = None


class Skill(BaseModel):
    """A skill with a proficiency rating."""

    name: Annotated[str, Field(description="Skill name, e.g. Python")]
    level: Annotated[
        Literal["beginner", "intermediate", "advanced", "expert"],
        Field(description="Proficiency level"),
    ]
    years_experience: Annotated[int, Field(description="Years of experience with this skill", ge=0)]


class Employee(BaseModel):
    """An employee in an organisation."""

    id: Annotated[str, Field(description="Unique employee ID")]
    full_name: Annotated[str, Field(description="Full name")]
    role: Annotated[str, Field(description="Job title")]
    contact: Annotated[ContactInfo, Field(description="Contact details")]
    skills: Annotated[list[Skill], Field(description="Skills this employee has")]
    reports_to_id: Annotated[str | None, Field(description="ID of their manager, or null for top-level")] = None


class Department(BaseModel):
    """A department within a company."""

    name: Annotated[str, Field(description="Department name")]
    cost_centre: Annotated[str, Field(description="Cost centre code")]
    head: Annotated[Employee, Field(description="Department head / director")]
    members: Annotated[list[Employee], Field(description="All employees in this department including the head")]
    annual_budget_usd: Annotated[float, Field(description="Annual budget in USD", gt=0)]


class GeoLocation(BaseModel):
    """Geographic coordinates."""

    latitude: Annotated[float, Field(description="Latitude in decimal degrees", ge=-90, le=90)]
    longitude: Annotated[float, Field(description="Longitude in decimal degrees", ge=-180, le=180)]


class Office(BaseModel):
    """A physical office location."""

    city: Annotated[str, Field(description="City name")]
    country_code: Annotated[str, Field(description="ISO 3166-1 alpha-2 country code", min_length=2, max_length=2)]
    address: Annotated[str, Field(description="Street address")]
    coordinates: Annotated[GeoLocation, Field(description="GPS coordinates of the office")]
    is_headquarters: Annotated[bool, Field(description="True if this is the company HQ")]


class Company(BaseModel):
    """A company with departments and offices."""

    name: Annotated[str, Field(description="Company name")]
    ticker: Annotated[str | None, Field(description="Stock ticker symbol, or null if private")] = None
    departments: Annotated[list[Department], Field(description="All departments", min_length=1)]
    offices: Annotated[list[Office], Field(description="All office locations", min_length=1)]
    founded_year: Annotated[int, Field(description="Year the company was founded", ge=1800)]

    @model_validator(mode="after")
    def must_have_exactly_one_hq(self) -> Company:
        hq_count = sum(1 for o in self.offices if o.is_headquarters)
        if hq_count != 1:
            raise ValueError(f"Company must have exactly one HQ, found {hq_count}")
        return self


# --- Order / e-commerce models ---


class ProductCategory(str, Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    SOFTWARE = "software"
    OTHER = "other"


class Product(BaseModel):
    """A product available for purchase."""

    sku: Annotated[str, Field(description="Stock-keeping unit identifier")]
    name: Annotated[str, Field(description="Product name")]
    category: Annotated[ProductCategory, Field(description="Product category")]
    unit_price_usd: Annotated[float, Field(description="Price in USD per unit", gt=0)]
    weight_kg: Annotated[float | None, Field(description="Weight in kilograms, or null for digital goods")] = None


class ShippingAddress(BaseModel):
    """A shipping destination."""

    recipient_name: Annotated[str, Field(description="Name of the recipient")]
    line1: Annotated[str, Field(description="Address line 1")]
    line2: Annotated[str | None, Field(description="Address line 2, or null")] = None
    city: Annotated[str, Field(description="City")]
    state_or_province: Annotated[str | None, Field(description="State or province, or null")] = None
    postal_code: Annotated[str, Field(description="Postal/ZIP code")]
    country_code: Annotated[str, Field(description="ISO 3166-1 alpha-2 country code", min_length=2, max_length=2)]


class OrderLine(BaseModel):
    """A single line in a customer order."""

    product: Annotated[Product, Field(description="The product being ordered")]
    quantity: Annotated[int, Field(description="Number of units ordered", ge=1)]
    discount_pct: Annotated[float, Field(description="Line-level discount percentage 0-100", ge=0, le=100)] = 0.0

    @property
    def line_total(self) -> float:
        return self.product.unit_price_usd * self.quantity * (1 - self.discount_pct / 100)


class CustomerTier(str, Enum):
    STANDARD = "standard"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class Customer(BaseModel):
    """A registered customer."""

    customer_id: Annotated[str, Field(description="Unique customer identifier")]
    full_name: Annotated[str, Field(description="Full name")]
    email: Annotated[str, Field(description="Email address")]
    tier: Annotated[CustomerTier, Field(description="Loyalty tier: standard, silver, gold, or platinum")]
    saved_addresses: Annotated[list[ShippingAddress], Field(description="Customer's saved shipping addresses")]


class Order(BaseModel):
    """A complete customer order."""

    order_id: Annotated[str, Field(description="Unique order identifier")]
    customer: Annotated[Customer, Field(description="The customer placing the order")]
    lines: Annotated[list[OrderLine], Field(description="Order lines, at least one", min_length=1)]
    shipping_address: Annotated[ShippingAddress, Field(description="Where to ship the order")]
    shipping_method: Annotated[
        Literal["standard", "express", "overnight"],
        Field(description="Shipping speed: standard, express, or overnight"),
    ]
    gift_message: Annotated[str | None, Field(description="Optional gift message")] = None

    @model_validator(mode="after")
    def shipping_address_must_have_country(self) -> Order:
        if not self.shipping_address.country_code:
            raise ValueError("Shipping address must include a country code")
        return self


# ---------------------------------------------------------------------------
# Test 9 – four-level hierarchy: Company → Department → Employee → Skill
# ---------------------------------------------------------------------------


class TestFourLevelOrganisationHierarchy:
    """Company → Department → Employee → ContactInfo + list[Skill]."""

    async def test_company_auto_coerced_at_all_four_levels(self):
        """All four hierarchy levels are auto-coerced to proper model instances.

        Company → Department → Employee → ContactInfo / list[Skill] and
        Company → list[Office] → GeoLocation are all coerced automatically,
        so every level supports attribute access without any manual validation.
        """
        received: list[object] = []

        def register_company(company: Company) -> str:
            """Register a company and all its departments and employees.

            Args:
                company: The full company record with departments, offices, and staff
            """
            received.append(company)
            return (
                f"Registered '{company.name}' with "
                f"{len(company.departments)} dept(s) and {len(company.offices)} office(s)"
            )

        result = await complete(
            user_message=(
                "Register a company called 'Acme Inc' founded in 2010, ticker ACME. "
                "It has one department 'Engineering' (cost centre ENG-01, budget $2M) "
                "headed by Alice Chen (ID emp-1, role CTO, email alice@acme.com, "
                "skills: Python expert 8 years, Kubernetes advanced 4 years). "
                "Alice reports to nobody. The department has one other member: "
                "Bob Ray (ID emp-2, role SWE, email bob@acme.com, "
                "skills: Python intermediate 2 years), who reports to emp-1. "
                "One office: HQ in San Francisco US at '1 Market St', "
                "coordinates 37.7749 lat -122.4194 lon. "
                "Use the register_company tool."
            ),
            system_prompt="You are a helpful assistant. Use the register_company tool when asked to register a company.",
            tools=[register_company],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        company = received[0]
        assert isinstance(company, Company), f"Expected Company, got {type(company).__name__}"

        # Level 1 → departments are Department instances
        assert len(company.departments) >= 1
        dept = company.departments[0]
        assert isinstance(dept, Department)

        # Level 2 → head is an Employee instance
        assert isinstance(dept.head, Employee)

        # Level 3 → contact info is a ContactInfo instance
        assert isinstance(dept.head.contact, ContactInfo)

        # Level 3 → skills are Skill instances
        assert all(isinstance(s, Skill) for s in dept.head.skills)

        # Level 1 → offices are Office instances
        assert len(company.offices) >= 1
        assert isinstance(company.offices[0], Office)

        # Level 2 → coordinates inside office is a GeoLocation instance
        assert isinstance(company.offices[0].coordinates, GeoLocation)



# ---------------------------------------------------------------------------
# Test 10 – e-commerce order: Customer → saved_addresses, lines → product
# ---------------------------------------------------------------------------


class TestEcommerceOrderComplexRelationships:
    """Order → Customer (with saved ShippingAddresses) → list[OrderLine] → Product."""

    async def test_order_auto_coerced_throughout_all_relationships(self):
        """Every nested object in Order is auto-coerced to the correct model type.

        Order → Customer (with list[ShippingAddress]) → list[OrderLine] → Product
        are all coerced, including enums (CustomerTier, ProductCategory), so
        the tool receives a fully populated object graph with attribute access.
        """
        received: list[object] = []

        def place_order(order: Order) -> str:
            """Place a customer order.

            Args:
                order: The complete order with customer, lines, and shipping details
            """
            received.append(order)
            return (
                f"Order {order.order_id} for "
                f"'{order.customer.full_name}' with {len(order.lines)} line(s)"
            )

        result = await complete(
            user_message=(
                "Place order ORD-999 for customer C-42 'Jane Doe' (jane@example.com, gold tier) "
                "who has one saved address: Jane Doe, 55 Baker St, London, postal SW1A 1AA, country GB. "
                "Order lines: 2x SKU 'LAPTOP-PRO' 'ProBook 15' electronics at $999.99 each (no discount), "
                "and 1x SKU 'BAG-01' 'Laptop Bag' other at $49.99 (10% discount). "
                "Ship to the saved address via express. No gift message. "
                "Use the place_order tool."
            ),
            system_prompt="You are a helpful assistant. Use the place_order tool when asked to place orders.",
            tools=[place_order],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        order = received[0]
        assert isinstance(order, Order), f"Expected Order, got {type(order).__name__}"

        # Customer is a Customer instance with enum tier
        assert isinstance(order.customer, Customer)
        assert isinstance(order.customer.tier, CustomerTier)

        # saved_addresses are ShippingAddress instances
        for addr in order.customer.saved_addresses:
            assert isinstance(addr, ShippingAddress)

        # Lines are OrderLine instances with coerced Product and enum category
        assert len(order.lines) >= 1
        for line in order.lines:
            assert isinstance(line, OrderLine)
            assert isinstance(line.product, Product)
            assert isinstance(line.product.category, ProductCategory)

    async def test_computed_property_on_order_lines_works(self):
        """The ``line_total`` computed property works because ``OrderLine`` is a proper instance.

        A property that accesses ``self.product.unit_price_usd`` would fail if
        ``product`` were still a plain dict — this confirms the full object graph
        is coerced and functional.
        """
        received: list[Order] = []

        def place_order_with_totals(order: Order) -> str:
            """Place a customer order and compute line totals.

            Args:
                order: The complete order to place
            """
            received.append(order)
            total = sum(line.line_total for line in order.lines)
            physical = [l for l in order.lines if l.product.weight_kg is not None]
            return (
                f"Order {order.order_id}: total=${total:.2f}, "
                f"ship_to={order.shipping_address.city}, "
                f"physical_items={len(physical)}, tier={order.customer.tier.value}"
            )

        result = await complete(
            user_message=(
                "Place order ORD-777 for customer C-99 'Mark Smith' (mark@shop.io, platinum tier) "
                "with one saved address: Mark Smith, 1 High Street, Edinburgh, postal EH1 1YZ, country GB. "
                "Order: 3x SKU 'TSHIRT-M' 'Classic Tee' clothing $19.99 weight 0.2kg (5% discount), "
                "and 1x SKU 'EBOOK-PY' 'Python Mastery' software $29.99 (no weight, digital). "
                "Ship to the saved address via standard. Gift message: 'Enjoy!'. "
                "Use place_order_with_totals."
            ),
            system_prompt="You are a helpful assistant. Use the place_order_with_totals tool when asked to place orders.",
            tools=[place_order_with_totals],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        order = received[0]
        assert isinstance(order, Order)
        assert isinstance(order.customer.tier, CustomerTier)
        assert isinstance(order.shipping_address, ShippingAddress)
        assert len(order.shipping_address.country_code) == 2
        for line in order.lines:
            assert isinstance(line, OrderLine)
            assert isinstance(line.product, Product)
            assert isinstance(line.product.category, ProductCategory)
        total = sum(line.line_total for line in order.lines)
        assert total > 0

    async def test_order_tool_composes_with_structured_output(self):
        """A Pydantic-annotated tool composes cleanly with ``structured_complete``.

        gluellm coerces the tool's input model AND the response_format model;
        both paths run independently without conflict.
        """
        from gluellm.api import structured_complete

        class OrderSummary(BaseModel):
            order_id: Annotated[str, Field(description="The order ID")]
            item_count: Annotated[int, Field(description="Total number of line items")]
            estimated_total_usd: Annotated[float, Field(description="Estimated order total in USD")]
            ships_to_country: Annotated[str, Field(description="2-letter country code of destination")]

        placed_orders: list[Order] = []

        def place_order_for_summary(order: Order) -> str:
            """Place an order and return key figures.

            Args:
                order: The complete order to place
            """
            placed_orders.append(order)
            total = sum(line.line_total for line in order.lines)
            return (
                f"order_id={order.order_id}, items={len(order.lines)}, "
                f"total={total:.2f}, country={order.shipping_address.country_code}"
            )

        result = await structured_complete(
            user_message=(
                "Place order ORD-555 for customer C-7 'Sara Lee' (sara@co.io, silver tier), "
                "no saved addresses. "
                "2x SKU 'MUG-01' 'Coffee Mug' other $12.50 weight 0.3kg. "
                "Ship to Sara Lee, 99 Pine Ave, Toronto, ON, postal M5V 2T6, country CA, via overnight. "
                "Use place_order_for_summary, then fill in the OrderSummary."
            ),
            system_prompt=(
                "You are a helpful assistant. Use the place_order_for_summary tool, "
                "then return a structured OrderSummary."
            ),
            tools=[place_order_for_summary],
            response_format=OrderSummary,
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(placed_orders) >= 1
        assert isinstance(placed_orders[0], Order)
        assert isinstance(placed_orders[0].customer, Customer)

        assert result.structured_output is not None
        summary = result.structured_output
        assert isinstance(summary, OrderSummary)
        assert summary.item_count >= 1
        assert summary.estimated_total_usd > 0
        assert len(summary.ships_to_country) == 2


# ---------------------------------------------------------------------------
# Helpers for dynamic model tests
# ---------------------------------------------------------------------------


def _make_tool_for_model(
    model_cls: type[BaseModel],
    tool_name: str,
    received_store: list,
) -> object:
    """Build a tool function whose single parameter is annotated with *model_cls*.

    The function is given *tool_name* as its ``__name__`` so gluellm can look
    it up by name, and a docstring whose ``Args`` section describes the
    parameter so the LLM schema picks up field descriptions from the model.
    """
    import inspect

    param = inspect.Parameter(
        "payload",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=model_cls,
    )
    sig = inspect.Signature([param], return_annotation=str)

    def _tool(payload):
        received_store.append(payload)
        return f"ok: received {type(payload).__name__}"

    _tool.__name__ = tool_name
    _tool.__qualname__ = tool_name
    _tool.__doc__ = (
        f"Process a {model_cls.__name__} payload.\n\n"
        f"Args:\n    payload: {model_cls.__doc__ or model_cls.__name__} instance"
    )
    _tool.__signature__ = sig
    _tool.__annotations__ = {"payload": model_cls, "return": str}
    return _tool


# ---------------------------------------------------------------------------
# Test 11 – pydantic.create_model() with flat fields
# ---------------------------------------------------------------------------


class TestDynamicFlatModel:
    """Tools whose parameter type is built with ``pydantic.create_model()`` at runtime."""

    async def test_flat_dynamic_model_auto_coerced_to_model_instance(self):
        """A tool annotated with a ``create_model`` type receives a model instance.

        any_llm coerces the JSON payload to the dynamically-created model class,
        so attribute access works even though the class was built at runtime.
        """
        ReportRequest = create_model(
            "ReportRequest",
            title=(Annotated[str, Field(description="Report title")], ...),
            period=(Annotated[str, Field(description="Reporting period, e.g. Q1 2025")], ...),
            include_charts=(
                Annotated[bool, Field(description="Whether to include charts")],
                True,
            ),
        )

        received: list[object] = []
        tool = _make_tool_for_model(ReportRequest, "generate_report", received)

        result = await complete(
            user_message=(
                "Generate a report titled 'Sales Overview' for period 'Q1 2025' "
                "with charts included using the generate_report tool."
            ),
            system_prompt="You are a helpful assistant. Use the generate_report tool when asked.",
            tools=[tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        payload = received[0]
        assert isinstance(payload, ReportRequest), (
            f"Expected ReportRequest instance, got {type(payload).__name__}"
        )
        assert isinstance(payload.title, str)  # type: ignore[union-attr]
        assert len(payload.title) > 0  # type: ignore[union-attr]



# ---------------------------------------------------------------------------
# Test 12 – create_model() with nested dynamic models
# ---------------------------------------------------------------------------


class TestDynamicNestedModels:
    """Dynamically created models that nest other dynamically created models."""

    async def test_nested_dynamic_models_auto_coerced_at_all_levels(self):
        """Nested ``create_model`` types are auto-coerced at every level.

        Route → list[Waypoint] → Coordinates are all dynamically created yet
        all coerced to proper instances, so attribute access works throughout.
        """
        Coordinates = create_model(
            "Coordinates",
            lat=(Annotated[float, Field(description="Latitude")], ...),
            lon=(Annotated[float, Field(description="Longitude")], ...),
        )
        Waypoint = create_model(
            "Waypoint",
            name=(Annotated[str, Field(description="Waypoint name")], ...),
            coordinates=(
                Annotated[Coordinates, Field(description="GPS coordinates")],
                ...,
            ),
            altitude_m=(
                Annotated[float | None, Field(description="Altitude in metres, or null")],
                None,
            ),
        )
        Route = create_model(
            "Route",
            route_id=(Annotated[str, Field(description="Unique route identifier")], ...),
            waypoints=(
                Annotated[
                    list[Waypoint],
                    Field(description="Ordered list of waypoints", min_length=2),
                ],
                ...,
            ),
            total_distance_km=(
                Annotated[float, Field(description="Total route distance in km", gt=0)],
                ...,
            ),
        )

        received: list[object] = []
        tool = _make_tool_for_model(Route, "create_route", received)

        result = await complete(
            user_message=(
                "Create route RT-01 with two waypoints: "
                "'Start' at lat 51.5 lon -0.1 altitude 10m, "
                "and 'End' at lat 51.6 lon -0.12 no altitude. "
                "Total distance 15.3km. Use the create_route tool."
            ),
            system_prompt="You are a helpful assistant. Use the create_route tool when asked.",
            tools=[tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        route = received[0]
        assert isinstance(route, Route), f"Expected Route, got {type(route).__name__}"

        waypoints = route.waypoints  # type: ignore[union-attr]
        assert isinstance(waypoints, list) and len(waypoints) >= 2

        wp = waypoints[0]
        assert isinstance(wp, Waypoint), f"Expected Waypoint, got {type(wp).__name__}"

        coords = wp.coordinates  # type: ignore[union-attr]
        assert isinstance(coords, Coordinates), (
            f"Expected Coordinates instance, got {type(coords).__name__}"
        )
        assert isinstance(coords.lat, float)  # type: ignore[union-attr]



# ---------------------------------------------------------------------------
# Test 13 – model built from a runtime schema (field names from config)
# ---------------------------------------------------------------------------


class TestRuntimeSchemaFromConfig:
    """Models whose field names are not known until runtime — built from a config dict."""

    async def test_tool_with_schema_driven_dynamic_model(self):
        """A model whose fields come from an external schema definition works end-to-end.

        This simulates a plugin system or form-builder where the schema is read
        from a database / config file at startup.
        """
        # Imagine this arrives from a database or config file at runtime
        form_schema = {
            "first_name": (Annotated[str, Field(description="First name of the applicant")], ...),
            "last_name": (Annotated[str, Field(description="Last name of the applicant")], ...),
            "age": (Annotated[int, Field(description="Applicant age in years", ge=18)], ...),
            "referral_code": (
                Annotated[str | None, Field(description="Optional referral code")],
                None,
            ),
        }

        ApplicationForm = create_model("ApplicationForm", **form_schema)

        received: list[object] = []
        tool = _make_tool_for_model(ApplicationForm, "submit_application", received)

        result = await complete(
            user_message=(
                "Submit an application for Jane Smith, age 29, "
                "referral code REF-XYZ. Use the submit_application tool."
            ),
            system_prompt="You are a helpful assistant. Use the submit_application tool when asked.",
            tools=[tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        form = received[0]
        assert isinstance(form, ApplicationForm)
        assert hasattr(form, "first_name") and isinstance(form.first_name, str)  # type: ignore[union-attr]
        assert hasattr(form, "age") and form.age >= 18  # type: ignore[union-attr]

    async def test_multiple_tools_each_with_distinct_dynamic_model(self):
        """Multiple tools registered simultaneously, each with its own dynamic model type."""
        EventModel = create_model(
            "CalendarEvent",
            title=(Annotated[str, Field(description="Event title")], ...),
            date=(Annotated[str, Field(description="Date in YYYY-MM-DD format")], ...),
            duration_minutes=(
                Annotated[int, Field(description="Duration in minutes", ge=1)],
                60,
            ),
        )
        ReminderModel = create_model(
            "Reminder",
            message=(Annotated[str, Field(description="Reminder message text")], ...),
            remind_at=(
                Annotated[str, Field(description="ISO 8601 datetime to trigger the reminder")],
                ...,
            ),
            repeat=(
                Annotated[
                    Literal["never", "daily", "weekly"],
                    Field(description="Repeat frequency: never, daily, or weekly"),
                ],
                "never",
            ),
        )

        events_received: list[object] = []
        reminders_received: list[object] = []

        event_tool = _make_tool_for_model(EventModel, "create_event", events_received)
        reminder_tool = _make_tool_for_model(ReminderModel, "set_reminder", reminders_received)

        result = await complete(
            user_message=(
                "Create a calendar event titled 'Team Standup' on 2026-04-15 for 30 minutes. "
                "Also set a weekly reminder 'Review metrics' at 2026-04-14T09:00:00. "
                "Use both create_event and set_reminder tools."
            ),
            system_prompt=(
                "You are a helpful assistant. "
                "Use create_event to schedule events and set_reminder to schedule reminders."
            ),
            tools=[event_tool, reminder_tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 2

        assert len(events_received) >= 1
        event = events_received[0]
        assert isinstance(event, EventModel)
        assert isinstance(event.title, str)  # type: ignore[union-attr]
        assert isinstance(event.duration_minutes, int)  # type: ignore[union-attr]

        assert len(reminders_received) >= 1
        reminder = reminders_received[0]
        assert isinstance(reminder, ReminderModel)
        assert isinstance(reminder.message, str)  # type: ignore[union-attr]
        assert reminder.repeat in ("never", "daily", "weekly")  # type: ignore[union-attr]

    async def test_dynamic_model_with_field_validator(self):
        """A validator added to a ``create_model`` type fires during model_validate."""

        def _normalise_currency(v: str) -> str:
            return v.strip().upper()

        CurrencyAmount = create_model(
            "CurrencyAmount",
            amount=(Annotated[float, Field(description="Monetary amount", gt=0)], ...),
            currency=(
                Annotated[str, Field(description="3-letter ISO 4217 currency code, e.g. usd")],
                ...,
            ),
            __validators__={
                "currency_normaliser": field_validator("currency")(_normalise_currency)
            },
        )

        received: list[object] = []
        tool = _make_tool_for_model(CurrencyAmount, "log_payment", received)

        result = await complete(
            user_message=(
                "Log a payment of 250.00 in 'usd' currency using the log_payment tool."
            ),
            system_prompt="You are a helpful assistant. Use the log_payment tool when asked.",
            tools=[tool],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1
        assert len(received) >= 1

        payment = received[0]
        assert isinstance(payment, CurrencyAmount)
        assert payment.currency == payment.currency.upper(), (  # type: ignore[union-attr]
            "field_validator should have uppercased the currency code"
        )


# ---------------------------------------------------------------------------
# Regression – AttributeError: 'dict' object has no attribute '<field>'
# ---------------------------------------------------------------------------


class TestDictAttributeErrorRegression:
    """Regression tests for the bug where tools annotated with Pydantic models
    received plain dicts and raised AttributeError on attribute access.

    The pattern below mirrors real-world finance tools that do ``arg.value``
    on a nested model like ``Quantity(value=5559, unit='USD million', sources=[...])``.
    """

    async def test_quantity_like_model_attribute_access_does_not_raise(self):
        """Tool with a Quantity-like Pydantic param receives a model, not a dict.

        Before the fix this raised::

            AttributeError: 'dict' object has no attribute 'value'

        because gluellm called ``tool_func(**json.loads(arguments))`` without
        coercing the dict to the annotated model type first.
        """

        attribute_errors: list[AttributeError] = []
        results: list[str] = []

        def report_quantity(qty: Quantity) -> str:
            """Report a financial quantity.

            Args:
                qty: The quantity to report, including value, unit and sources
            """
            try:
                # This is the natural usage that used to raise AttributeError
                formatted = f"{qty.value} {qty.unit}"
                results.append(formatted)
                return formatted
            except AttributeError as exc:
                attribute_errors.append(exc)
                return f"AttributeError: {exc}"

        result = await complete(
            user_message=(
                "Report a quantity of 5559 USD million using the report_quantity tool. "
                "Include one source with id 'doc-1' and quote 'revenue was 5559 USD million'."
            ),
            system_prompt="You are a helpful assistant. Use the report_quantity tool when asked.",
            tools=[report_quantity],
        )

        assert isinstance(result, ExecutionResult)
        assert result.tool_calls_made >= 1

        assert len(attribute_errors) == 0, (
            f"Got AttributeError on qty.value — dict was passed instead of Quantity: "
            f"{attribute_errors[0]}"
        )
        assert len(results) >= 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 0
