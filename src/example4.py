import os 
import asyncio
import duckdb
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai import ModelSettings
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

llm_provider = OpenAIChatModel(
    'google/gemma-4-e4b', 
    provider=OpenAIProvider(
        base_url='http://127.0.0.1:1234/v1',         
        api_key='test'
    ),
)

max_token_settings = ModelSettings(max_tokens=8912)

agent = Agent(
llm_provider,
    model_settings=max_token_settings, 
    instructions=(
        "You are a helpful assistant agent. "
        "Always try and give accurate answers and ask questions to the user if you are not sure. "
    ),
)


@agent.tool_plain
def add_values(x: float, y: float) -> float:
    """Add x to y and return the result"""
    print(f"Tool called: add_values {x} to {y}")
    return x + y

@agent.tool_plain
def what_time_is_it() -> str:
    print(f"Agent is looking at its watch...")
    return datetime.now().strftime("%H:%M:%S")


async def main():
    await agent.to_cli()

if __name__ == "__main__":
    asyncio.run(main())


