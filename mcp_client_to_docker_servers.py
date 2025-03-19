# /// script
# dependencies = [
#   "anthropic",
#   "mcp",
#   "click",
# ]
# ///

"""
MCP Client for connecting to Docker-based MCP servers.

This script is taken from "https://modelcontextprotocol.io/quickstart/client". Updated it
to work with MCP docker servers.

To use this, you need to have the following installed:
    - Docker
    - uv
    - "ANTHROPIC_API_KEY" set

Sample run
==========
uv run \
    "https://raw.githubusercontent.com/ryandam9/uv_tools/refs/heads/master/mcp_client_to_docker_servers.py" \
    mcp/time-zone-converter
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Optional

import click
from anthropic import Anthropic
from anthropic.types import ToolParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.write = None
        self.stdio = None
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, docker_image) -> None:
        """Connect to an MCP server

        Args:
            docker_image: MCP server docker image name
        """
        command = "docker"
        args = ["run", "-i", "--rm", docker_image]

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str, model: str, max_tokens: int) -> str:
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()

        available_tools = [
            ToolParam(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
            )
            for tool in response.tools
        ]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)

            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = dict(content.input)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(
                    f"[Calling tool '{tool_name}' with args '{tool_args}']"
                )

                # Continue conversation with tool results
                if hasattr(content, "text") and content.text:
                    messages.append({"role": "assistant", "content": content.text})
                messages.append({"role": "user", "content": result.content})

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self, model, max_tokens):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query, model, max_tokens)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run_client(docker_image, model, max_tokens):
    """Run the MCP client with the specified Docker image."""
    client = MCPClient()
    try:
        await client.connect_to_server(docker_image)
        await client.chat_loop(model, max_tokens)
    finally:
        await client.cleanup()


@click.command()
@click.argument(
    "docker_image",
    required=True,
)
@click.option(
    "--model",
    default="claude-3-7-sonnet-20250219",
    help="Claude model to use (default: claude-3-7-sonnet-20250219)",
)
@click.option(
    "--max-tokens",
    default=1024,
    type=int,
    help="Maximum tokens for Claude response (default: 1024)",
)
def main(docker_image, model, max_tokens):
    """MCP Client for connecting to Docker-based MCP servers.

    This client connects to an MCP server running in a Docker container and
    provides an interactive chat interface with tool-calling capabilities.

    DOCKER_IMAGE: Name of the Docker image containing the MCP server to connect to.
    """
    asyncio.run(run_client(docker_image, model, max_tokens))


if __name__ == "__main__":
    main()
