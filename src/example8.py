import os 
import asyncio
import duckdb
from datetime import datetime
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai import ModelSettings
from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai.capabilities import WebSearch

llm_provider = OpenAIChatModel(
    'google/gemma-4-e4b', 
    provider=OpenAIProvider(
        base_url='http://127.0.0.1:1234/v1',         
        api_key='test'
    ),
)

max_token_settings = ModelSettings(max_tokens=8912)

context_manager = ContextManagerCapability(
    max_tokens=max_token_settings["max_tokens"],
    summarization_model=llm_provider  
)


@dataclass
class Deps:
    pass

manager_agent =  Agent(
    llm_provider,
    model_settings=max_token_settings, 
    instructions=(
        "You are a manager agent. "
        "Your job is to figure out what the user wants then to make use of your companion agents"
        " to answer their question or solve their problem as best you can. "
    ),
    capabilities=[
        context_manager, 
    ],  
)

weather_agent = Agent(
    llm_provider,
    model_settings=max_token_settings, 
    instructions=(
        "You are a weather agent. "
        "Given a location, you will look up the current weather forecast."
        "You do not like long answers - you will return a succinct description of the weather. "   
    ),
    capabilities=[
        WebSearch(local='duckduckgo'), 
    ],  
)

fashion_agent = Agent(
   llm_provider, 
    deps_type=Deps,
    output_type=str,
    instructions=(
        "You are a fashionista fashion assistant. "
        "Given a weather forecast, recommend what to wear. "
        "Be concise. You are very opinionated and have a great personality. "
        "You were educated in the finest fashion houses in Milan, Paris and Tokyo"
        "Everyone loves hearing your  opinions about the latest trends"
    ),
)

@manager_agent.tool
async def weather_agent_tool(ctx: RunContext[Deps], location: str) -> str:
    """ Call out to the weather agent, sending it the location and it will return the weather 
        forecast for that location
    """
    print(f"Manager calling weather agent: with location {location}")
    result = await weather_agent.run(
        f"Get weather for {location}",
        deps=ctx.deps,
        usage=ctx.usage,  # important for tracking usage across agents
    )
    return result.output


@manager_agent.tool
async def fashion_agent_tool(ctx: RunContext[Deps], person_information:str, forecast: str) -> str:
    """Get some fashion advice from the fashion agent given  """
    print(f"Calling the fashion agent for person '{person_information}' and weather '{forecast}'")
    result = await fashion_agent.run(
        f"Person with info {person_information } wants fashion advice. The weather is: {forecast}. Please assist us!",
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return result.output


# @agent.tool_plain
# def add_values(x: float, y: float) -> float:
#     """Add x to y and return the result"""
#     print(f"Tool called: add_values {x} to {y}")
#     return x + y

# @agent.tool_plain
# def what_time_is_it() -> str:
#     return datetime.now().strftime("%H:%M:%S")


async def main():
    await manager_agent.to_cli()

if __name__ == "__main__":
    asyncio.run(main())


