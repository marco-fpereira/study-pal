from config.mcp_config import MCP_SERVERS
from model.enum.mcp_tool_enum import MCPToolsEnum
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

class MCPRepository:
    def __init__(
        self,
        server_name: MCPToolsEnum
    ):
        self.server_name = server_name
        self.client = MultiServerMCPClient(
            connections={self.server_name.value: MCP_SERVERS[self.server_name.value]}
        )


    async def list_tools(self) -> list[BaseTool]: 
        """
        Retrieve a list of available tools from the connected MCP server.

        Returns:
            List of Tool objects representing the available tools on the server.
        """

        return await self.client.get_tools(server_name=self.server_name.value)


    async def call_tool(self, tool_name: str, arguments: dict) -> ToolMessage:
        """
        Invoke a specific tool on the MCP server with the given arguments.

        Args:
            tool_name: str - The name of the tool to invoke.
            arguments: dict - A dictionary of arguments to pass to the tool.

        Returns:
            ToolMessage containing the result of the tool invocation.
        """
        tools = await self.client.get_tools(server_name=self.server_name.value)
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found on server '{self.server_name.value}'")
        return await tool.ainvoke(arguments)
        

    async def close(self): 
        """
        Cleanly close the connection to the MCP server.

        Returns:
            None
        """
        if self.client:
            self._context_manager.__aexit__(None, None, None)
            self.client = None