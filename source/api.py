"""GlueLLM Python API - High-level interface for LLM interactions."""

import json
from typing import TypeVar, Type, Callable, Optional, Any
from pydantic import BaseModel, Field
from typing import Annotated

from any_llm import completion as any_llm_completion
from any_llm.types.completion import ChatCompletion

from source.models.config import RequestConfig
from source.models.prompt import SystemPrompt
from source.models.conversation import Role, Conversation

T = TypeVar("T", bound=BaseModel)


class ToolExecutionResult(BaseModel):
    """Result of a tool execution loop."""
    final_response: Annotated[str, Field(description="The final text response from the model")]
    tool_calls_made: Annotated[int, Field(description="Number of tool calls made")]
    tool_execution_history: Annotated[list[dict[str, Any]], Field(description="History of tool calls and results")]
    raw_response: Annotated[ChatCompletion, Field(description="The raw final response from the LLM")]


class GlueLLM:
    """High-level API for LLM interactions with automatic tool execution."""
    
    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        system_prompt: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
        max_tool_iterations: int = 10,
    ):
        """Initialize GlueLLM client.
        
        Args:
            model: Model identifier in format "provider:model_name"
            system_prompt: System prompt content (defaults to helpful assistant)
            tools: List of callable functions to use as tools
            max_tool_iterations: Maximum number of tool call iterations (prevents infinite loops)
        """
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tools = tools or []
        self.max_tool_iterations = max_tool_iterations
        self._conversation = Conversation()
    
    def complete(
        self,
        user_message: str,
        execute_tools: bool = True,
    ) -> ToolExecutionResult:
        """Complete a request with automatic tool execution loop.
        
        Args:
            user_message: The user's message/request
            execute_tools: Whether to automatically execute tools and loop
            
        Returns:
            ToolExecutionResult with final response and execution history
        """
        # Add user message to conversation
        self._conversation.add_message(Role.USER, user_message)
        
        # Build initial messages
        system_message = {
            "role": "system",
            "content": self._format_system_prompt(),
        }
        messages = [system_message] + self._conversation.messages_dict
        
        tool_execution_history = []
        tool_calls_made = 0
        
        # Tool execution loop
        for iteration in range(self.max_tool_iterations):
            response = any_llm_completion(
                messages=messages,
                model=self.model,
                tools=self.tools if self.tools else None,
            )
            
            # Check if model wants to call tools
            if execute_tools and self.tools and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                
                # Add assistant message with tool calls to history
                messages.append(response.choices[0].message)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_calls_made += 1
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Find and execute the tool
                    tool_func = self._find_tool(tool_name)
                    if tool_func:
                        try:
                            tool_result = tool_func(**tool_args)
                            tool_result_str = str(tool_result)
                        except Exception as e:
                            tool_result_str = f"Error executing tool: {str(e)}"
                        
                        # Record in history
                        tool_execution_history.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result_str,
                        })
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_str,
                        })
                    else:
                        # Tool not found
                        error_msg = f"Tool '{tool_name}' not found"
                        tool_execution_history.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "result": error_msg,
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg,
                        })
                
                # Continue loop to get next response
                continue
            
            # No more tool calls, we have final response
            final_content = response.choices[0].message.content or ""
            
            # Add assistant response to conversation
            self._conversation.add_message(Role.ASSISTANT, final_content)
            
            return ToolExecutionResult(
                final_response=final_content,
                tool_calls_made=tool_calls_made,
                tool_execution_history=tool_execution_history,
                raw_response=response,
            )
        
        # Max iterations reached
        final_content = "Maximum tool execution iterations reached."
        return ToolExecutionResult(
            final_response=final_content,
            tool_calls_made=tool_calls_made,
            tool_execution_history=tool_execution_history,
            raw_response=response,
        )
    
    def structured_complete(
        self,
        user_message: str,
        response_format: Type[T],
    ) -> T:
        """Complete a request and return structured output.
        
        Args:
            user_message: The user's message/request
            response_format: Pydantic model class for structured output
            
        Returns:
            Instance of response_format with parsed data
        """
        # Add user message to conversation
        self._conversation.add_message(Role.USER, user_message)
        
        # Build messages
        system_message = {
            "role": "system",
            "content": self._format_system_prompt(),
        }
        messages = [system_message] + self._conversation.messages_dict
        
        # Call with response_format (no tools for structured output)
        response = any_llm_completion(
            messages=messages,
            model=self.model,
            response_format=response_format,
        )
        
        # Parse the response
        parsed = response.choices[0].message.parsed
        content = response.choices[0].message.content
        
        # Add assistant response to conversation
        if content:
            self._conversation.add_message(Role.ASSISTANT, content)
        
        # If parsed is already a Pydantic instance, return it
        # Otherwise, instantiate the model from the parsed dict/JSON
        if isinstance(parsed, response_format):
            return parsed
        
        # Handle case where parsed is a dict
        if isinstance(parsed, dict):
            return response_format(**parsed)
        
        # Fallback: try to parse from JSON string in content
        if content:
            import json
            try:
                data = json.loads(content)
                return response_format(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Last resort: return parsed as-is and hope for the best
        return parsed
    
    def _format_system_prompt(self) -> str:
        """Format system prompt with tools if available."""
        from source.models.prompt import BASE_SYSTEM_PROMPT
        return BASE_SYSTEM_PROMPT.render(
            instructions=self.system_prompt,
            tools=self.tools,
        ).strip()
    
    def _find_tool(self, tool_name: str) -> Optional[Callable]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.__name__ == tool_name:
                return tool
        return None
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._conversation = Conversation()


# Convenience functions for one-off requests

def complete(
    user_message: str,
    model: str = "openai:gpt-4o-mini",
    system_prompt: Optional[str] = None,
    tools: Optional[list[Callable]] = None,
    execute_tools: bool = True,
    max_tool_iterations: int = 10,
) -> ToolExecutionResult:
    """Quick completion with automatic tool execution.
    
    Args:
        user_message: The user's message/request
        model: Model identifier in format "provider:model_name"
        system_prompt: System prompt content
        tools: List of callable functions to use as tools
        execute_tools: Whether to automatically execute tools
        max_tool_iterations: Maximum number of tool call iterations
        
    Returns:
        ToolExecutionResult with final response and execution history
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        max_tool_iterations=max_tool_iterations,
    )
    return client.complete(user_message, execute_tools=execute_tools)


def structured_complete(
    user_message: str,
    response_format: Type[T],
    model: str = "openai:gpt-4o-mini",
    system_prompt: Optional[str] = None,
) -> T:
    """Quick structured completion.
    
    Args:
        user_message: The user's message/request
        response_format: Pydantic model class for structured output
        model: Model identifier in format "provider:model_name"
        system_prompt: System prompt content
        
    Returns:
        Instance of response_format with parsed data
    """
    client = GlueLLM(
        model=model,
        system_prompt=system_prompt,
    )
    return client.structured_complete(user_message, response_format)

