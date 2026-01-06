"""Production chatbot example using GlueLLM.

This example demonstrates building a customer support chatbot with:
- Multi-turn conversation management
- Tool integration for external APIs
- Error handling and fallbacks
- Structured responses
"""

import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from gluellm.api import GlueLLM


# Database simulation (in production, use real database)
class CustomerDB:
    """Simulated customer database."""

    def __init__(self):
        self.customers = {
            "12345": {"name": "Alice Smith", "email": "alice@example.com", "plan": "premium"},
            "67890": {"name": "Bob Jones", "email": "bob@example.com", "plan": "basic"},
        }

    def get_customer(self, customer_id: str) -> dict | None:
        """Get customer information by ID."""
        return self.customers.get(customer_id)

    def update_plan(self, customer_id: str, new_plan: str) -> bool:
        """Update customer plan."""
        if customer_id in self.customers:
            self.customers[customer_id]["plan"] = new_plan
            return True
        return False


# Initialize database
db = CustomerDB()


# Tool functions for the chatbot
def get_customer_info(customer_id: str) -> str:
    """Get customer information by ID.

    Args:
        customer_id: The customer ID to look up

    Returns:
        Customer information as a formatted string
    """
    customer = db.get_customer(customer_id)
    if customer:
        return f"Customer {customer['name']} ({customer['email']}) is on the {customer['plan']} plan."
    return f"Customer ID {customer_id} not found."


def update_customer_plan(customer_id: str, new_plan: str) -> str:
    """Update a customer's plan.

    Args:
        customer_id: The customer ID
        new_plan: The new plan name (basic, premium, enterprise)

    Returns:
        Confirmation message
    """
    if db.update_plan(customer_id, new_plan):
        return f"Successfully updated customer {customer_id} to {new_plan} plan."
    return f"Failed to update customer {customer_id}. Customer not found."


def check_system_status() -> str:
    """Check system status.

    Returns:
        Current system status
    """
    return "All systems operational. Uptime: 99.9%"


# Structured response model for ticket creation
class SupportTicket(BaseModel):
    """Support ticket information."""

    customer_id: Annotated[str, Field(description="Customer ID")]
    issue_type: Annotated[str, Field(description="Type of issue (billing, technical, account)")]
    priority: Annotated[str, Field(description="Priority level (low, medium, high, urgent)")]
    description: Annotated[str, Field(description="Detailed description of the issue")]


async def chatbot_example():
    """Run a customer support chatbot example."""
    print("=" * 60)
    print("Production Chatbot Example")
    print("=" * 60)

    # Create chatbot client with tools
    chatbot = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="""You are a helpful customer support agent.
        You can:
        - Look up customer information using get_customer_info
        - Update customer plans using update_customer_plan
        - Check system status using check_system_status

        Be friendly, professional, and helpful. Always confirm actions before executing them.""",
        tools=[get_customer_info, update_customer_plan, check_system_status],
        max_tool_iterations=5,
    )

    # Conversation flow
    conversations = [
        ("Hello! I need help with my account.", "greeting"),
        ("My customer ID is 12345. Can you tell me about my account?", "lookup"),
        ("I'd like to upgrade to premium plan.", "upgrade"),
        ("What's the current system status?", "status"),
    ]

    for user_message, _ in conversations:
        print(f"\n[User] {user_message}")
        try:
            result = await chatbot.complete(user_message)
            print(f"[Bot] {result.final_response}")
            if result.tool_calls_made > 0:
                print(f"  (Used {result.tool_calls_made} tool(s))")
        except Exception as e:
            print(f"[Error] {e}")
            # In production, log error and provide fallback response
            print("[Bot] I apologize, but I'm experiencing technical difficulties. Please try again.")

    # Example: Create support ticket with structured output
    print("\n" + "=" * 60)
    print("Creating Support Ticket (Structured Output)")
    print("=" * 60)

    ticket_client = GlueLLM(
        model="openai:gpt-4o-mini",
        system_prompt="Extract support ticket information from user messages. Be thorough and accurate.",
    )

    ticket_message = """
    I'm customer 12345 and I'm having a billing issue.
    I was charged twice this month and it's urgent that this gets resolved.
    Please help me fix this billing problem.
    """

    try:
        result = await ticket_client.structured_complete(ticket_message, response_format=SupportTicket)
        ticket = result.structured_output
        print("\nExtracted Ticket:")
        print(f"  Customer ID: {ticket.customer_id}")
        print(f"  Issue Type: {ticket.issue_type}")
        print(f"  Priority: {ticket.priority}")
        print(f"  Description: {ticket.description}")
    except Exception as e:
        print(f"Error creating ticket: {e}")

    print("\n" + "=" * 60)
    print("Chatbot Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(chatbot_example())
