#!/usr/bin/env python
# coding: utf-8

# # Multi-Agent Framework (Gemini + AutoGen)
# 
# This notebook serves as a standalone framework for running multi-agent systems using Google's Gemini models. It allows for flexible orchestration (Parallel or Round Robin) and easy modification of agents.
# 
# **Structure:**
# 1.  **Setup**: Environment and API Key configuration.
# 2.  **Model**: Initialization of the Gemini model client.
# 3.  **Agents**: Definition of specialized agents (Planner, Local, Language, Summary).
# 4.  **Orchestration**: Functions to run the agents in different patterns.
# 5.  **Execution**: Define the task and run the system.

# In[1]:


import os
import re
import sys
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (crucial for running asyncio in notebooks)
nest_asyncio.apply()

# --- API Key Setup ---
# 1. Try environment variable
# 2. Try looking for 'list_models.py' in the current directory (legacy support)
# 3. Fail if not found

if not os.getenv("GOOGLE_API_KEY"):
    try:
        if os.path.exists("list_models.py"):
            with open("list_models.py", "r") as f:
                txt = f.read()
            m = re.search(r"api_key\s*=\s*[\"']([^\"']+)[\"']", txt)
            if m:
                os.environ["GOOGLE_API_KEY"] = m.group(1)
                print("Loaded GOOGLE_API_KEY from list_models.py")
    except Exception as e:
        print(f"Warning: Could not read list_models.py: {e}")

if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set. Please set it in the environment variables or provide it here.")
    # os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" # Uncomment and set if needed
else:
    print("GOOGLE_API_KEY is set.")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


# In[2]:


# --- Model Configuration ---

model_name = "models/gemini-2.5-flash"

# Minimal model_info for AutoGen to identify Gemini capabilities
model_info = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.GEMINI_2_5_FLASH,
    "structured_output": True,
    "multiple_system_messages": True,
}

try:
    print(f"Initializing model: {model_name}")
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info=model_info,
    )
    print("Model client initialized successfully.")
except Exception as e:
    print(f"Failed to create model client: {e}")
    model_client = None


# In[ ]:


# --- Agent Definitions ---

if model_client:
    planner_agent = AssistantAgent(
        "planner_agent",
        model_client=model_client,
        description="A helpful assistant that can plan trips.",
        system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    )

    local_agent = AssistantAgent(
        "local_agent",
        model_client=model_client,
        description="A local assistant that can suggest local activities or places to visit.",
        system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    )

    language_agent = AssistantAgent(
        "language_agent",
        model_client=model_client,
        description="A helpful assistant that can provide language tips for a given destination.",
        system_message=(
            "You are a helpful assistant that can review travel plans, providing feedback on important/critical tips "
            "about how best to address language or communication challenges for the given destination."
        ),
    )

    # Base Summary Agent (Used for Parallel Mode where it just synthesizes provided text)
    travel_summary_agent_parallel = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message=(
            "You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. "
            "You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN."
        ),
    )

    # Interactive Summary Agent (Used for Round Robin / Dynamic Modes where it participates and terminates)
    travel_summary_agent_interactive = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message=(
            "You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. "
            "You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. "
            "When the plan is complete and all perspectives are integrated, you can respond with TERMINATE."
        ),
    )

    print("Agents initialized.")
else:
    print("Model client invalid, skipping agent initialization.")


# In[ ]:


# --- Orchestration Logic ---
from autogen_agentchat.messages import TextMessage

async def process_stream(stream, filename: str):
    """
    Consumes the agent stream, prints to console, and writes to a file.
    """
    print(f"--- Output will be saved to {filename} ---")
    with open(filename, "w", encoding="utf-8") as f:
        async for message in stream:
            # Basic formatting for the log/console
            output = ""
            if isinstance(message, TextMessage):
                output = f"\n[{message.source}]: {message.content}\n"
            else:
                output = f"\n[{message.source}]: {str(message)}\n"

            print(output)
            f.write(output)


async def run_parallel_team(task: str, specialist_agents: list, summary_agent: AssistantAgent, filename="output_parallel.txt"):
    """
    Runs specialist agents in parallel and aggregates their results for a summary agent.
    """
    print(f"Starting PARALLEL run for task: {task[:50]}...")

    # 1. Run specialists in parallel
    print("Dispatching tasks to specialists...")
    results = await asyncio.gather(*[agent.run(task=task) for agent in specialist_agents])

    # 2. Collect responses
    collected_context = []
    for res in results:
        if res.messages:
            last_msg = res.messages[-1]
            collected_context.append(f"--- {last_msg.source} suggestions ---\n{last_msg.content}")

    aggregated_info = "\n\n".join(collected_context)
    print("Parallel agents finished. Generating summary...")

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"--- PARALLEL AGENT INPUTS ---\n{aggregated_info}\n-----------------------------\n")

    # 3. Final Summary
    summary_task = (
        f"Original Request: {task}\n\n"
        f"Below are suggestions from your team:\n\n{aggregated_info}\n\n"
        "Please compile these into a cohesive response."
    )

    await process_stream(summary_agent.run_stream(task=summary_task), filename)


async def run_round_robin_team(task: str, participants: list, termination_word="TERMINATE", filename="output_round_robin.txt"):
    """
    Runs agents in a sequential round-robin group chat until termination.
    Agents speak in a fixed order: A -> B -> C -> A...
    """
    print(f"Starting ROUND ROBIN run for task: {task[:50]}...")

    termination = TextMentionTermination(termination_word)
    team = RoundRobinGroupChat(
        participants=participants,
        termination_condition=termination,
    )

    await process_stream(team.run_stream(task=task), filename)


async def run_dynamic_router_team(task: str, participants: list, model_client, termination_word="TERMINATE", filename="output_dynamic.txt"):
    """
    Runs agents in a dynamic group chat where the model selects the next speaker.
    Best for complex or non-linear tasks.
    """
    print(f"Starting DYNAMIC ROUTER run for task: {task[:50]}...")

    termination = TextMentionTermination(termination_word)

    # selector_prompt is optional, but helps guide the router
    selector_prompt = (
        "Select the next agent to speak based on the conversation history. "
        "If the plan is complete and agreed upon, select the summary agent to finalize, or Terminate if done."
    )

    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client, # The model is needed here to make selection decisions
        termination_condition=termination,
        selector_prompt=selector_prompt
    )

    await process_stream(team.run_stream(task=task), filename)


# In[ ]:


# --- Main Execution ---

async def main_execution():
    TASK = "Plan a 4-day trip to Mexico City. Provide suggestions for activities, places to visit, and language tips."

    if model_client:
        print(">>> MODE 1: PARALLEL execution running...")
        await run_parallel_team(
            task=TASK, 
            specialist_agents=[planner_agent, local_agent, language_agent], 
            summary_agent=travel_summary_agent_parallel,
            filename="output_mode_a_parallel.txt"
        )
        print(">>> MODE 1: DONE. Output saved to output_mode_a_parallel.txt\n")

        print(">>> MODE 2: ROUND ROBIN execution running...")
        # Note: Using interactive summary agent that knows to TERMINATE
        await run_round_robin_team(
            task=TASK,
            participants=[planner_agent, local_agent, language_agent, travel_summary_agent_interactive],
            filename="output_mode_b_round_robin.txt"
        )
        print(">>> MODE 2: DONE. Output saved to output_mode_b_round_robin.txt\n")

        print(">>> MODE 3: DYNAMIC ROUTER execution running...")
        await run_dynamic_router_team(
            task=TASK,
            participants=[planner_agent, local_agent, language_agent, travel_summary_agent_interactive],
            model_client=model_client,
            filename="output_mode_c_dynamic.txt"
        )
        print(">>> MODE 3: DONE. Output saved to output_mode_c_dynamic.txt\n")

    else:
        print("Model client invalid. Check API Key.")

if __name__ == "__main__":
    asyncio.run(main_execution())

