from fastmcp import FastMCP
from datetime import datetime
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

@mcp.resource("resource://timenow")
def get_date_time_now() -> str:
    """Get current date and time."""
    return datetime.now().isoformat()



if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)