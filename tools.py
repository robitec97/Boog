"""
Tool framework for Boog agent.

Provides base Tool class and implementations for web search, file operations.
All file operations are sandboxed to WORKSPACE_DIR for security.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path

# Import from app.py
try:
    import requests
    _HAVE_REQUESTS = True
except ModuleNotFoundError:
    import urllib.request, urllib.error
    _HAVE_REQUESTS = False


# Configure workspace directory
WORKSPACE_DIR = Path(os.getenv("BOOG_WORKSPACE_DIR", "/tmp/boog_workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

logger = logging.getLogger(__name__)


# ---------- HTTP Helper ----------
def _post_json(url: str, headers: dict, payload: dict) -> dict:
    """HTTP POST with JSON payload. Falls back to urllib if requests unavailable."""
    if _HAVE_REQUESTS:
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        r.raise_for_status()
        return r.json()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=12) as resp:  # nosec
        return json.loads(resp.read().decode("utf-8"))


# ---------- Base Tool Class ----------
class Tool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM function calling."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return result as string."""
        pass

    def to_function_definition(self) -> Dict[str, Any]:
        """Convert to Groq/OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ---------- Web Search Tool ----------
class WebSearchTool(Tool):
    """Search the web using Tavily API."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information, news, facts, or answers to questions. Returns relevant search results with URLs and snippets."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-8)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, num_results: int = 5) -> str:
        """Execute web search and return formatted results."""
        try:
            api_key = os.getenv("TAVILY_API_KEY", "")
            if not api_key:
                return "Error: Web search is not configured (TAVILY_API_KEY not set)."

            # Trim query to 400 chars
            q = query.strip()[:400]
            if not q:
                return "Error: Empty search query."

            # Call Tavily API
            payload = {
                "query": q,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "max_results": max(1, min(num_results, 8)),
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            data = _post_json("https://api.tavily.com/search", headers, payload)

            # Format results
            results = []
            for i, r in enumerate(data.get("results", [])[:num_results], 1):
                title = r.get("title", "").strip() or "Untitled"
                url = r.get("url", "")
                snippet = r.get("content", "").strip()[:500]
                results.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}")

            if not results:
                return f"No results found for query: {q}"

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            return f"Error: Web search failed - {str(e)}"


# ---------- File Operations Tools ----------
def _validate_path(file_path: str) -> Path:
    """
    Validate and resolve file path within workspace.
    Raises ValueError if path is outside workspace or invalid.
    """
    try:
        # Resolve path relative to workspace
        full_path = (WORKSPACE_DIR / file_path).resolve()

        # Ensure path is within workspace (prevent path traversal)
        if not str(full_path).startswith(str(WORKSPACE_DIR)):
            raise ValueError("Access denied: path outside workspace")

        return full_path
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")


class ReadFileTool(Tool):
    """Read file contents from workspace."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a text file in your workspace. Use this to examine files the user mentions or that would help answer their question."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace root)"
                }
            },
            "required": ["file_path"]
        }

    def execute(self, file_path: str) -> str:
        """Read file and return contents."""
        try:
            full_path = _validate_path(file_path)

            if not full_path.exists():
                return f"Error: File not found: {file_path}"

            if not full_path.is_file():
                return f"Error: Not a file: {file_path}"

            # Check file size
            size = full_path.stat().st_size
            if size > MAX_FILE_SIZE:
                return f"Error: File too large to read ({size} bytes, max {MAX_FILE_SIZE} bytes)"

            # Read file
            content = full_path.read_text(encoding='utf-8')

            return f"Contents of {file_path}:\n\n{content}"

        except ValueError as e:
            return f"Error: {str(e)}"
        except UnicodeDecodeError:
            return f"Error: Cannot read {file_path} - file is not text or uses unsupported encoding"
        except Exception as e:
            logger.error(f"Read file error: {e}", exc_info=True)
            return f"Error: Failed to read file - {str(e)}"


class WriteFileTool(Tool):
    """Write content to a file in workspace."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file in your workspace. Creates the file if it doesn't exist, overwrites if it does. Parent directories are created automatically."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to workspace root)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }

    def execute(self, file_path: str, content: str) -> str:
        """Write content to file."""
        try:
            full_path = _validate_path(file_path)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Check content size
            content_size = len(content.encode('utf-8'))
            if content_size > MAX_FILE_SIZE:
                return f"Error: Content too large ({content_size} bytes, max {MAX_FILE_SIZE} bytes)"

            # Write file
            full_path.write_text(content, encoding='utf-8')

            return f"Successfully wrote {len(content)} characters to {file_path}"

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Write file error: {e}", exc_info=True)
            return f"Error: Failed to write file - {str(e)}"


class ListFilesTool(Tool):
    """List files and directories in workspace."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files and directories in your workspace. Use this to explore what files are available or to navigate the directory structure."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list (relative to workspace root). Use '.' or omit for root.",
                    "default": "."
                }
            },
            "required": []
        }

    def execute(self, directory: str = ".") -> str:
        """List directory contents."""
        try:
            full_path = _validate_path(directory)

            if not full_path.exists():
                return f"Error: Directory not found: {directory}"

            if not full_path.is_dir():
                return f"Error: Not a directory: {directory}"

            # List contents
            items = []
            for item in sorted(full_path.iterdir()):
                rel_path = item.relative_to(WORKSPACE_DIR)
                if item.is_dir():
                    items.append(f"{rel_path}/")
                else:
                    size = item.stat().st_size
                    items.append(f"{rel_path} ({size:,} bytes)")

            if not items:
                return f"Directory {directory} is empty"

            return f"Contents of {directory}:\n" + "\n".join(items)

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"List files error: {e}", exc_info=True)
            return f"Error: Failed to list directory - {str(e)}"


# ---------- Tool Registry ----------
class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool:
        """Get tool by name."""
        return self.tools.get(name)

    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all tools as function definitions for Groq."""
        return [tool.to_function_definition() for tool in self.tools.values()]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments."""
        tool = self.get_tool(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = tool.execute(**arguments)
            logger.info(f"Tool {name} completed")
            return result
        except TypeError as e:
            return f"Error: Invalid arguments for {name} - {str(e)}"
        except Exception as e:
            logger.error(f"Tool execution error: {name}", exc_info=True)
            return f"Error: Tool {name} failed - {str(e)}"


# ---------- Initialize Default Tools ----------
def create_default_registry() -> ToolRegistry:
    """Create and populate tool registry with default tools."""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListFilesTool())
    return registry
