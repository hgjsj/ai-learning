from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import model as m
import asyncio


async def main():
    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {
                    "X-Custom-Header": "custom-value"
                },
            }
        }
    )

    tools = await mcp_client.get_tools(server_name="weather")
    resources = await mcp_client.get_resources(server_name="weather")


    for blob in resources:
        print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
        print(blob.as_string())

    model = m.get_gemini_model()

    agent = create_agent(
        model=model,
        tools=tools,
    )

    response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and msg.artifact:
            print(msg.artifact["structured_content"])
        else:
            msg.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())