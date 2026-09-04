# Multi-agent system workshop 

## 0. Get the code for the workshop

There are two ways to do this - you can go here:

https://github.com/yeeking/multi-agent-workshop

Click the green 'code' button and select 'download zip'. Unzip the zip somewhere on your filesystem. 

Or if you have git installed, you can rnu this command in your terminal:

```
git clone https://github.com/yeeking/multi-agent-workshop.git
```

### £nd of stage check-in

At this point, you should have the zip file downloaded, or the git repo cloned. You should see a folder on your computer entitled 'multi-agent-workshop'. 

## 1. Set up LM Studio

### Install LM Studio

Visit here and download LM studio: https://lmstudio.ai/download

You *do not* need 'Bionic' you need 'LM Studio'. 

### Download the Gemma model

Launch LM Studio and on the left panel click the model search function. Search for 'google/gemma-4-e4b' - it is quite small and capable. Or if you are on a memory limited system (e.g. an 8GB Mac) or you want a faster download, you should look for 'https://lmstudio.ai/models/google/gemma-4-e2b' which is smaller but still useful. 

Once the download completes, go into the chat section in LM Studio, start a new chat and load the google/gemma-4-e4b model. Try typing in some questions  to verify the model is working correctly. 

### About speed

The chat interface of LM-Studio should display some statistics about the performance of the model on your system. On my regular laptop it says I am achieved 18 tokens per second. That means it generates about 18 'words' or 'word fragments' per second. But you will see that it spends some of the 'tokens' thinking, before giving you the final output. How many tokens per second do you see on your system? 

### Start the API server

Now we are going to set up LM Studio so we can make calls to the models from Python. This involves starting the LM Studio API server.

Go to the developer icon in LM Studio. Select the gemma model and start up the server. 

It should display a message saying 'Status: running' next to the server on/ off toggle button. 

### Test the API server with your web browser

Point your web browser here: http://127.0.0.1:1234/v1/models

You should see some JSON data displayed which describes a list of models. 

### End of stage check

Check you have done the following before proceeding:

* Have VSCodium or VSCode installed and running
* opened the repo folder in the code editor
* installed LM-Studio 
* installed a Gemma model
* chatted to Gemma in LM-Studio
* started the LM-Studio API server
* tested the API server in your web browser

## 2. Create a minimal Pydantic AI agent

Now we have our language model server up and running, we are going to try to communicate with it from Python. We will not do this at a low level - instead we will use the Pydantic-AI Python library. 

### Install vscodium

If you do not have VSCode or VSCodium installed, please install it now. 

We recommend VSCodium as it has stronger privacy features. Go here and download for your operating system: 

https://vscodium.com/

Launch VSCodium once installed and select 'Open folder' from the file menu. 

Select the folder where you downloaded the multi-agent github codebase. 

### Install Pydantic AI

Earlier, you cloned the multi-agent-workshop repo. Open up VSCODIUM or VSCODE and point it at the repo folder. 

In the code editor terminal, browse to the repo folder. Run these commands to create a virtual environment and to install the necessary python packages there. 

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install pydantic-ai duckdb summarization-pydantic-ai "pydantic-ai-slim[duckduckgo]"

```

### Create a minimal agent

Now create a new file src/test.py with the following code in it:

```
import os 
import asyncio
import duckdb

from pydantic_ai import Agent
from pydantic_ai import ModelSettings
from pydantic_ai_summarization import ContextManagerCapability
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

res = agent.run_sync('What is the population of Singapore?')
print(res)
```
Run that code - make sure you run it inside the venv you just created. 

You can find that code in src/example1.py


### Try a command-line interface

Swap out the last lines:

```
res = agent.run_sync('What is the population of Singapore?')
print(res)
```

For these:

```
async def main():
    await agent.to_cli()

if __name__ == "__main__":
    asyncio.run(main())
```

Run again and you can interact with the agent via a command line interface. This is the most reliable interactive interface. See src/example2.py

### Try the web interface

Swap out the code:
```
async def main():
    await agent.to_cli()

if __name__ == "__main__":
    asyncio.run(main())
```

For:

```
app = agent.to_web()
```

To run this one, in the Terminal, type:

```
cd src
uvicorn example3:app
```
You should see some logs like:

```
INFO:     Started server process [215031]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Now in your web browser, go to http://localhost:8000/

You should see a web interface wherein you can talk to your agent. 

Try some prompts. Congratulations - you now have your own private large language model. 

### End of stage check-in

* Made and activated a virtual env
* Installed pydantic and other packages
* Run an example 
* Interacted with an agent in the CLI
* Interacted with an agent in your web browser

## 3. Add some tools

### Give it calculation capabilities

Chatting to an agent is all very well, but it cannot see or act in the outside world and it has to use inference to answer all questions. By giving the agent tools, we can give it some more concrete, reliable abilities. Let's start with a very simple addition tool. Building on either your web UI or CLI example, add this code below the line where you create the agent (see example4.py):

```
@agent.tool_plain
def add_values(x: float, y: float) -> float:
    """Add x to y and return the result"""
    print(f"Tool called: add_values {x} to {y}")
    return x + y
```

Try asking it to add some values together. Here is an example:

```
pydantic-ai ➤ what is 10.2552 + 2135.12312?
Tool called: add_values 10.2552 to 2135.12312
The sum of 10.2552 and 2135.12312 is 2145.37832.                                                
```

Any ideas why it might be useful to give the model an adding tool that directly carries out the computation rather than letting the model do it? 

Try this tool:

```
@agent.tool_plain
def what_time_is_it() -> str:
    return datetime.now().strftime("%H:%M:%S")
```

### How does tool calling work? 

Let's dig into the details of how the tool calling works. 

You can investigate tool calling by observing the logs on LM Studio - you can do this via the GUI or you can use the lms command-line tool:

```
lms logs stream
```

I see some logs like this:

```
timestamp: 8/30/2026, 4:18:22 PM
type: llm.prediction.input
modelIdentifier: google/gemma-4-e4b
modelPath: google/gemma-4-e4b
input:
<bos><|turn>system
<|think|>
You are a helpful assistant agent. Always try and give accurate answers and ask questions to the user if you are not sure.<|tool>declaration:add_values{description:<|"|>Add x to y and return the result<|"|>,parameters:{properties:{x:{type:<|"|>NUMBER<|"|>},y:{type:<|"|>NUMBER<|"|>}},required:[<|"|>x<|"|>,<|"|>y<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
what is 10.2552 + 2135.12312?<turn|>
<|turn>model
<|tool_call>call:add_values{x:10.2552,y:2135.12312}<tool_call|><|tool_response>response:add_values{value:<|"|>2145.3783200000003<|"|>}<tool_response|><turn|>
The sum of 10.2552 and 2135.12312 is **2145.37832**.<turn|>
<|turn>user
how about 12321.25125 + 1536.636366 + 2132.213123?<turn|>
<|turn>model
<|tool_call>call:add_values{x:12321.25125,y:1536.636366}<tool_call|><|tool_response>response:add_values{value:<|"|>13857.887616<|"|>}<tool_response|><turn|>
<|tool_call>call:add_values{x:13857.887616,y:2132.213123}<tool_call|><|tool_response>response:add_values{value:<|"|>15990.100739<|"|>}<tool_response|><turn|>
The sum of 12321.25125, 1536.636366, and 2132.213123 is **15990.100739**.<turn|>
<|turn>user
what is 2+2?<turn|>
<|turn>model



timestamp: 8/30/2026, 4:18:22 PM
type: llm.prediction.output
modelIdentifier: google/gemma-4-e4b
output:
<|channel>thought
The user is asking a simple arithmetic question: "what is 2+2?". I can use the `add_values` tool to calculate this, or simply answer it as it's common knowledge. Since the request is extremely straightforward and doesn't involve complex numbers or previous context that would require multiple steps, a direct answer is appropriate, but using the tool maintains consistency with the demonstrated behavior of solving math problems via tools when applicable.

Plan:
1. Call `add_values` with x=2 and y=2.
2. Return the result to the user.<channel|><|tool_call>call:add_values{x:2,y:2}<tool_call|>


timestamp: 8/30/2026, 4:18:23 PM
type: llm.prediction.input
modelIdentifier: google/gemma-4-e4b
modelPath: google/gemma-4-e4b
input:
<bos><|turn>system
<|think|>
You are a helpful assistant agent. Always try and give accurate answers and ask questions to the user if you are not sure.<|tool>declaration:add_values{description:<|"|>Add x to y and return the result<|"|>,parameters:{properties:{x:{type:<|"|>NUMBER<|"|>},y:{type:<|"|>NUMBER<|"|>}},required:[<|"|>x<|"|>,<|"|>y<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
what is 10.2552 + 2135.12312?<turn|>
<|turn>model
<|tool_call>call:add_values{x:10.2552,y:2135.12312}<tool_call|><|tool_response>response:add_values{value:<|"|>2145.3783200000003<|"|>}<tool_response|><turn|>
The sum of 10.2552 and 2135.12312 is **2145.37832**.<turn|>
<|turn>user
how about 12321.25125 + 1536.636366 + 2132.213123?<turn|>
<|turn>model
<|tool_call>call:add_values{x:12321.25125,y:1536.636366}<tool_call|><|tool_response>response:add_values{value:<|"|>13857.887616<|"|>}<tool_response|><turn|>
<|tool_call>call:add_values{x:13857.887616,y:2132.213123}<tool_call|><|tool_response>response:add_values{value:<|"|>15990.100739<|"|>}<tool_response|><turn|>
The sum of 12321.25125, 1536.636366, and 2132.213123 is **15990.100739**.<turn|>
<|turn>user
what is 2+2?<turn|>
<|turn>model
<|tool_call>call:add_values{x:2,y:2}<tool_call|><|tool_response>response:add_values{value:<|"|>4.0<|"|>}<tool_response|><turn|>
<|turn>model



timestamp: 8/30/2026, 4:18:23 PM
type: llm.prediction.output
modelIdentifier: google/gemma-4-e4b
output:
The sum of 2 and 2 is **4**.
```

If you read through those logs you can see that the pydantic AI library is adding a whole load more text into the actual prompt I sent ('what is 2+2?'). It sends my agent instructions, it sends a description of the tool. Pydantic-AI also 'unpacks' the response from the model, which includes interpreting the requested tool calls, and actually making those tool calls. 

Can you identify how the agent sends the request for tool calls back? 

### End of stage check-in

* Added an 'addition' tool to the agent
* Created your own tool and added it to the agent
* Investigated how tool-calling works 

## 4. Add some agent capabilities

As well as custom tools, Pydantic-AI provides some useful capabilities. We could implement these ourselves, but why not use the built-in ones?

### Search capability

Web search allows the agent to look things up on the internet. 

Add this line to the top of the script (example5.py)

```
from pydantic_ai.capabilities import WebSearch
```

Then adjust your agent spec as follows:

```
agent = Agent(
    llm_provider,
    model_settings=max_token_settings, 
    instructions=(
        "You are a helpful assistant agent. "
        "Always try and give accurate answers and ask questions to the user if you are not sure. "
    ),
    capabilities=[
        WebSearch(local='duckduckgo')
    ],  
)

```

Try asking it for some recent news with and without web search. Try asking it for the weather!  

### Add context management

One problem with language models is that the prompt has a maximum length. For longer sequences of chat interactions, the context gets longer and longer as the complete chat history gets passed into the model each time. A solution to this is the Context Manager Capability (example6.py). 

Add this: 

```
from pydantic_ai_summarization import ContextManagerCapability

... then the agent...

agent = Agent(
    llm_provider,
    model_settings=max_token_settings, 
    instructions=(
        "You are a helpful assistant agent. "
        "Always try and give accurate answers and ask questions to the user if you are not sure. "
    ),
    capabilities=[
        context_manager, 
        WebSearch(local='duckduckgo'), 
    ],  
)
```

You should be able to continue chatting away now! 

### End of stage check-in

* Added an internet search capability to an agent
* Added context summarisation capability  to an agent 

## 5. Multi-agent system

Now we are going to convert our single agent setup into a multi-agent setup with a purpose.  

### The 'What to wear' problem

We'll start with the slightly playful case study of a fashion advisor system (example7.py).

First of all, check out example7.py. You will see that there are two agents specified: manager_agent and weather_agent. 

```
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
```

Note that only the weather agent has access to web search. 

We then attach the weather agent to the manager, as a tool with specified communication:

```
@manager_agent.tool
async def weather_agent_tool(ctx: RunContext[Deps], location: str) -> str:
    # inside the function we just call run on the weather agent 
    print(f"Manager calling weather agent: with location {str}")
    result = await weather_agent.run(
        f"Get weather for {location}",
        deps=ctx.deps,
        usage=ctx.usage,  # important for tracking usage across agents
    )
    return result.output
```

If you read carefully you will see that the weather agent is provided as a tool to the manager agent, and the tool receives a ctx object (don't worry too much about that for now) plus a location string. It returns a string. It then constructs a prompt for the weather agent and returns the result. 

Why separate concerns like this? 

* management of context: the weather agent can operate efficiently with minimal context. Short context (prompt length) means potentially faster responses. 
* different models: you could potentially use a very basic, small model for the weather agent and a more complex planner for the manager agent
* personalities and approach: by sending different 'instructions' to the agents, you can have team members with different approaches and different personalities. For example - I have made the weather agent succinct, but the manager agent is 'helpful', 
* separate contexts: since each agent manages its own memory, you can control what is visible or known. 

Try asking some questions about the weather to the manager agent. See if it calls out the weather agent. You might find the web ui more helpful to see how the agents interact. 

### Add another agent

Now let's add another agent - the fashionista. The manager agent will use this agent for advice on what to wear. 

```
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


```

Now try out some questions! 

Can you think of any other kind of agents that you might want to use? Or a more interested scenario? Experiment with teams of agents with different tools and capabilities. 

### End of stage check-in

* Made an agent team (multi-agent system)
* Used the team to make fashion suggestions based on the weather 