from pathlib import Path

current_file = Path(__file__).resolve()
base_path = current_file.parent.parent.parent
server_path = base_path / "infra" / "mcp-server"

# MCP server config & client setup
MCP_SERVERS = {
    "question_generator": {
        "transport": "streamable-http",
        "url": "http://localhost:8000/mcp"
    }
    # Future MCP servers can be added here with the same structure
}
