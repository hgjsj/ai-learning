import model as m
from datetime import datetime, timedelta
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import  ModelRequest, ModelResponse


SYSTEM_PROMPT = """
You are an expert weather forecaster, who speaks in puns.

You have access to tools:

- get_weather_for_location: use this to get the weather for a specific location and date
- get_date_time_now: use this to get the date for a specific day (like "now", "tomorrow", "yesterday")
- get_user_location: use this to get the user's location if not specified

If a user asks you for the weather, make sure you know the location and the time. 
If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.
If the user mentions a time (like tomorrow), use get_date_time_now to get the date string.
"""
Checkpoint = InMemorySaver()

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str
    # user location is optional; if not provided, the agent can use it
    user_location: str

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

@tool
def get_weather_for_location(city: str, t:str) -> str:
    """Get weather for a given city."""
    print(f"The weather in {city} is sunny in {t}.")
    return f"The weather in {city} is sunny in {t}."

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    return runtime.context.user_location

@tool
def get_date_time_now(day: str) -> str:
    """Get the time based on the day."""
    current_time = datetime.now()
    if "tomorrow" in day.lower():
        target_time = current_time + timedelta(days=1)
    elif "yesterday" in day.lower():
        target_time = current_time - timedelta(days=1)
    else:
        target_time = current_time

    return target_time.strftime("%Y-%m-%d")

@wrap_tool_call
def handle_tool_call(request, handler):
    """Middleware to handle tool calls with context."""
    try:
        return handler(request)
    except Exception as e:
        tool_name = request.tool_call["name"]
        return ToolMessage(
            content=f"Tool {tool_name} errored with message: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )


#model = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature= 0)
model = m.get_gemini_model()

agent = create_agent(model=model,
                     system_prompt=SYSTEM_PROMPT,
                     tools=[get_weather_for_location, get_date_time_now, get_user_location],
                     context_schema=Context,
                     response_format=ToolStrategy(ResponseFormat),
                     middleware=[handle_tool_call],
                     checkpointer=Checkpoint)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather here tomorrow?"}]},
    config=config,
    context=Context(user_id="1", user_location="Florida")
)
print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1", user_location="Florida")
)

print(response['structured_response'])