# ← NEW: Tool discovery, invocation, result parsing
from model.enum.mcp_tool_enum import MCPToolsEnum
from repository.mcp_repository import MCPRepository

class MCPService:
    def __init__(self, server_name: MCPToolsEnum):
        self.repo = MCPRepository(server_name=server_name)


    async def invoke(self, tool_name: str, args: dict) -> str: 
        result = await self.repo.call_tool(tool_name=tool_name, arguments=args)
        return result

    async def get_available_tools(self) -> list[dict]:
        return await self.repo.list_tools()


