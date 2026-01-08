"""
Agent orchestration logic for Boog.

Handles tool execution loop, response generation, and streaming.
"""

import json
import logging
import re
from typing import Dict, List, Any, Generator, Tuple, Optional
from groq import Groq

from prompts import build_system_prompt
from session import Conversation
from tools import ToolRegistry


logger = logging.getLogger(__name__)

# Maximum tool execution iterations to prevent infinite loops
MAX_ITERATIONS = 5


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking/reasoning sections from response.

    Looks for patterns like "Let me think...", "Breaking this down:", etc.

    Args:
        text: Response text to parse

    Returns:
        Tuple of (thinking_content, cleaned_text)
    """
    if not text:
        return None, text

    thinking_patterns = [
        r"(?:Let me think|Let me consider|Breaking this down|Here's my thinking)[:.]?\s*(.*?)(?=\n\n|\Z)",
        r"Thinking[:.]?\s*(.*?)(?=\n\n|\Z)",
    ]

    thinking_parts = []
    cleaned_text = text

    for pattern in thinking_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            thinking_content = match.group(0).strip()
            if len(thinking_content) > 20:  # Only capture substantial thinking
                thinking_parts.append(thinking_content)
                cleaned_text = cleaned_text.replace(match.group(0), "").strip()

    thinking = "\n".join(thinking_parts) if thinking_parts else None
    return thinking, cleaned_text


def generate_agent_response(
    conversation: Conversation,
    tool_registry: ToolRegistry,
    groq_client: Groq
) -> Dict[str, Any]:
    """
    Generate agent response with tool execution loop (non-streaming).

    Args:
        conversation: Current conversation
        tool_registry: Available tools
        groq_client: Groq API client

    Returns:
        Dict with response, steps, session_id, conversation_id
    """
    steps = []
    executed_tool_calls = set()  # Track (tool_name, args_hash) to prevent duplicates

    # Build messages with system prompt
    messages = [
        {"role": "system", "content": build_system_prompt()},
        *conversation.get_messages()
    ]

    try:
        # Tool execution loop
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Agent iteration {iteration + 1}")

            # Call Groq with tools
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tool_registry.get_function_definitions(),
                tool_choice="auto",
                temperature=0.6,
            )

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                logger.info(f"Agent requested {len(message.tool_calls)} tool calls")

                # Add assistant message with tool calls to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Track if any tools were actually executed this iteration
                tools_executed_this_iteration = False

                # Execute each tool
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments for {tool_name}")
                        arguments = {}

                    # Create a key for de-duplication
                    try:
                        args_key = json.dumps(arguments, sort_keys=True)
                    except (TypeError, ValueError):
                        args_key = str(arguments)
                    tool_key = (tool_name, args_key)

                    # Skip duplicate tool calls
                    if tool_key in executed_tool_calls:
                        logger.warning(f"Skipping duplicate tool call: {tool_name}")
                        # Still need to add a placeholder result for the API
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "(Already executed this tool with same arguments)"
                        })
                        continue

                    executed_tool_calls.add(tool_key)
                    tools_executed_this_iteration = True

                    # Record tool call step
                    steps.append({
                        "type": "tool_call",
                        "tool": tool_name,
                        "arguments": arguments
                    })

                    # Execute tool
                    result = tool_registry.execute_tool(tool_name, arguments)

                    # Limit result size to prevent message bloat
                    if len(result) > 10000:
                        result = result[:10000] + "\n... (truncated)"

                    # Record tool result step
                    steps.append({
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": result
                    })

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # If no new tools were executed, break to prevent infinite loop
                if not tools_executed_this_iteration:
                    logger.warning("No new tools executed, breaking loop")
                    break

                # Continue loop to get synthesis
                continue

            # No more tool calls - final response
            final_response = message.content or ""

            # Extract thinking if present
            thinking, response_text = extract_thinking(final_response)
            if thinking:
                steps.append({"type": "thinking", "content": thinking})

            # Add final response step
            steps.append({"type": "response", "content": response_text})

            # Save to conversation
            conversation.add_message("assistant", response_text)

            return {
                "response": response_text,
                "steps": steps,
                "session_id": conversation.session_id,
                "conversation_id": conversation.id
            }

        # Max iterations reached
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached")
        error_msg = "I apologize, but I got caught in a loop. Let's try a different approach."

        conversation.add_message("assistant", error_msg)

        return {
            "response": error_msg,
            "steps": steps,
            "session_id": conversation.session_id,
            "conversation_id": conversation.id
        }

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        error_msg = f"I encountered an error: {str(e)}"

        return {
            "response": error_msg,
            "steps": steps,
            "session_id": conversation.session_id,
            "conversation_id": conversation.id,
            "error": str(e)
        }


def stream_agent_response(
    conversation: Conversation,
    tool_registry: ToolRegistry,
    groq_client: Groq
) -> Generator[str, None, None]:
    """
    Generate agent response with streaming (Server-Sent Events).

    Yields SSE-formatted messages for each step.

    Args:
        conversation: Current conversation
        tool_registry: Available tools
        groq_client: Groq API client

    Yields:
        SSE formatted strings
    """
    # Build messages with system prompt
    messages = [
        {"role": "system", "content": build_system_prompt()},
        *conversation.get_messages()
    ]

    final_response = ""
    executed_tool_calls = set()  # Track (tool_name, args_hash) to prevent duplicates

    try:
        # Tool execution loop
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Agent streaming iteration {iteration + 1}")

            # Call Groq with streaming
            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tool_registry.get_function_definitions(),
                tool_choice="auto",
                temperature=0.6,
                stream=True,
            )

            tool_calls = []
            tool_calls_dict = {}  # Index -> tool call data
            content_buffer = ""

            # Process streaming chunks
            for chunk in stream:
                delta = chunk.choices[0].delta

                # Stream content
                if delta.content:
                    content_buffer += delta.content
                    # Send content delta
                    yield f"data: {json.dumps({'type': 'content_delta', 'content': delta.content})}\n\n"

                # Collect tool calls
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        idx = tool_call_chunk.index

                        # Initialize tool call if new
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tool_call_chunk.id or "",
                                "function": {
                                    "name": tool_call_chunk.function.name or "",
                                    "arguments": ""
                                }
                            }

                        # Append arguments
                        if tool_call_chunk.function.arguments:
                            tool_calls_dict[idx]["function"]["arguments"] += tool_call_chunk.function.arguments

            # Convert tool_calls_dict to list
            if tool_calls_dict:
                tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]

            # Execute tool calls if present
            if tool_calls:
                logger.info(f"Agent requested {len(tool_calls)} tool calls")

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": content_buffer,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                # Track if any tools were actually executed this iteration
                tools_executed_this_iteration = False

                # Execute each tool
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments for {tool_name}")
                        arguments = {}

                    # Create a key for de-duplication
                    try:
                        args_key = json.dumps(arguments, sort_keys=True)
                    except (TypeError, ValueError):
                        args_key = str(arguments)
                    tool_key = (tool_name, args_key)

                    # Skip duplicate tool calls
                    if tool_key in executed_tool_calls:
                        logger.warning(f"Skipping duplicate tool call: {tool_name}")
                        # Still need to add a placeholder result for the API
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "(Already executed this tool with same arguments)"
                        })
                        continue

                    executed_tool_calls.add(tool_key)
                    tools_executed_this_iteration = True

                    # Send tool call event
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'arguments': arguments})}\n\n"

                    # Execute tool
                    result = tool_registry.execute_tool(tool_name, arguments)

                    # Limit result size to prevent message bloat
                    if len(result) > 10000:
                        result = result[:10000] + "\n... (truncated)"

                    # Send tool result event
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': result})}\n\n"

                    # Add to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })

                # If no new tools were executed, break to prevent infinite loop
                if not tools_executed_this_iteration:
                    logger.warning("No new tools executed, breaking loop")
                    break

                # Continue to next iteration
                continue

            # No tool calls - done
            final_response = content_buffer

            # Extract thinking if present
            thinking, response_text = extract_thinking(final_response)
            if thinking:
                yield f"data: {json.dumps({'type': 'thinking', 'content': thinking})}\n\n"

            # Save to conversation
            conversation.add_message("assistant", response_text)

            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'session_id': conversation.session_id, 'conversation_id': conversation.id})}\n\n"
            return

        # Max iterations reached
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached in streaming")
        error_msg = "I apologize, but I got caught in a loop. Let's try a different approach."

        conversation.add_message("assistant", error_msg)

        yield f"data: {json.dumps({'type': 'content_delta', 'content': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Agent streaming error: {e}", exc_info=True)
        error_msg = f"I encountered an error: {str(e)}"

        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
