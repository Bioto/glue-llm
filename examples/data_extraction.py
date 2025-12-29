"""Data extraction system example using GlueLLM.

This example demonstrates:
- Structured data extraction from unstructured text
- Multiple extraction patterns
- Validation and error handling
- Batch processing
"""

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from gluellm.api import structured_complete


# Data models for extraction
class Product(BaseModel):
    """Product information."""

    name: Annotated[str, Field(description="Product name")]
    price: Annotated[float, Field(description="Price in USD")]
    category: Annotated[str, Field(description="Product category")]
    in_stock: Annotated[bool, Field(description="Whether product is in stock")]


class Contact(BaseModel):
    """Contact information."""

    name: Annotated[str, Field(description="Full name")]
    email: Annotated[str, Field(description="Email address")]
    phone: Annotated[str | None, Field(description="Phone number", default=None)]
    company: Annotated[str | None, Field(description="Company name", default=None)]


class Event(BaseModel):
    """Event information."""

    title: Annotated[str, Field(description="Event title")]
    date: Annotated[str, Field(description="Event date")]
    location: Annotated[str, Field(description="Event location")]
    attendees: Annotated[list[str], Field(description="List of attendee names")]


class Invoice(BaseModel):
    """Invoice information."""

    invoice_number: Annotated[str, Field(description="Invoice number")]
    date: Annotated[str, Field(description="Invoice date")]
    total_amount: Annotated[float, Field(description="Total amount in USD")]
    items: Annotated[list[str], Field(description="List of invoice items")]
    customer_name: Annotated[str, Field(description="Customer name")]


async def data_extraction_example():
    """Run data extraction examples."""
    print("=" * 60)
    print("Data Extraction System Example")
    print("=" * 60)

    # Example 1: Product extraction
    print("\n1. Product Extraction")
    print("-" * 60)
    product_text = """
    Product: iPhone 15 Pro Max
    Price: $1,199.00
    Category: Electronics
    Status: In Stock
    """
    try:
        product = await structured_complete(
            user_message=f"Extract product information:\n{product_text}",
            response_format=Product,
            system_prompt="Extract product information accurately. Convert prices to float.",
        )
        print(f"Name: {product.name}")
        print(f"Price: ${product.price:.2f}")
        print(f"Category: {product.category}")
        print(f"In Stock: {product.in_stock}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Contact extraction
    print("\n2. Contact Extraction")
    print("-" * 60)
    contact_text = """
    Name: Sarah Johnson
    Email: sarah.johnson@techcorp.com
    Phone: (555) 123-4567
    Company: TechCorp Inc.
    """
    try:
        contact = await structured_complete(
            user_message=f"Extract contact information:\n{contact_text}",
            response_format=Contact,
            system_prompt="Extract contact details accurately.",
        )
        print(f"Name: {contact.name}")
        print(f"Email: {contact.email}")
        print(f"Phone: {contact.phone}")
        print(f"Company: {contact.company}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Event extraction
    print("\n3. Event Extraction")
    print("-" * 60)
    event_text = """
    Tech Conference 2024
    Date: March 15, 2024
    Location: San Francisco Convention Center
    Attendees: John Smith, Jane Doe, Bob Wilson, Alice Brown
    """
    try:
        event = await structured_complete(
            user_message=f"Extract event information:\n{event_text}",
            response_format=Event,
            system_prompt="Extract event details. Parse attendee list as array.",
        )
        print(f"Title: {event.title}")
        print(f"Date: {event.date}")
        print(f"Location: {event.location}")
        print(f"Attendees: {', '.join(event.attendees)}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Invoice extraction
    print("\n4. Invoice Extraction")
    print("-" * 60)
    invoice_text = """
    Invoice #INV-2024-001
    Date: January 15, 2024
    Customer: Acme Corporation
    Items:
    - Web Development Services: $5,000.00
    - Consulting Hours: $2,500.00
    - Hosting Setup: $500.00
    Total: $8,000.00
    """
    try:
        invoice = await structured_complete(
            user_message=f"Extract invoice information:\n{invoice_text}",
            response_format=Invoice,
            system_prompt="Extract invoice details. Parse items as array of strings.",
        )
        print(f"Invoice Number: {invoice.invoice_number}")
        print(f"Date: {invoice.date}")
        print(f"Customer: {invoice.customer_name}")
        print(f"Total: ${invoice.total_amount:.2f}")
        print(f"Items: {len(invoice.items)}")
        for item in invoice.items:
            print(f"  - {item}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 5: Batch processing
    print("\n5. Batch Processing")
    print("-" * 60)
    products_text = [
        "MacBook Pro, $2,499, Computers, In Stock",
        "AirPods Pro, $249, Audio, In Stock",
        "iPad Air, $599, Tablets, Out of Stock",
    ]

    print("Processing multiple products...")
    extracted_products = []
    for text in products_text:
        try:
            product = await structured_complete(
                user_message=f"Extract product: {text}",
                response_format=Product,
                system_prompt="Extract product information. Parse 'In Stock' as True, 'Out of Stock' as False.",
            )
            extracted_products.append(product)
        except Exception as e:
            print(f"  Error extracting {text}: {e}")

    print(f"\nExtracted {len(extracted_products)} products:")
    for product in extracted_products:
        print(f"  - {product.name}: ${product.price:.2f} ({product.category})")

    print("\n" + "=" * 60)
    print("Data Extraction Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(data_extraction_example())
