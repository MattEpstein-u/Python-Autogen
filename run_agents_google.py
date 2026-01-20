import os
import re
import sys
import asyncio

# Try to set GOOGLE_API_KEY from environment or fall back to reading list_models.py
if not os.getenv("GOOGLE_API_KEY"):
    try:
        with open("list_models.py", "r") as f:
            txt = f.read()
        m = re.search(r"api_key\s*=\s*[\"']([^\"']+)[\"']", txt)
        if m:
            os.environ["GOOGLE_API_KEY"] = m.group(1)
    except FileNotFoundError:
        pass

if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set and not found in list_models.py")
    sys.exit(1)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

async def main():
    # Configure Gemini 2.5 Flash model
    model_name = "models/gemini-2.5-flash"
    
    # Provide minimal model_info so AutoGen knows these are Gemini models
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
    except Exception as e:
        print(f"Failed to create model client: {e}")
        return

    # Define the specialist agents
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

    # Define the summary agent
    travel_summary_agent = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message=(
            "You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. "
            "You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN."
        ),
    )

    task = "Plan a 5-day trip to Mexico City. Provide suggestions for activities, places to visit, and language tips."

    print("Starting agents run in parallel (this may take a few seconds)...")
    try:
        # Run specialist agents in parallel
        # We use .run() for the specialists to get their full response primarily
        print("dispatching tasks to specialists...")
        results = await asyncio.gather(
            planner_agent.run(task=task),
            local_agent.run(task=task),
            language_agent.run(task=task)
        )

        # Collect their responses
        collected_context = []
        for res in results:
            if res.messages:
                last_msg = res.messages[-1]
                collected_context.append(f"--- {last_msg.source} suggestions ---\n{last_msg.content}")

        aggregated_info = "\n\n".join(collected_context)
        
        print("Parallel agents finished. Generating summary...")

        # Pass the collected information to the summary agent for the final plan
        summary_task = (
            f"Original Request: {task}\n\n"
            f"Below are suggestions from your team:\n\n{aggregated_info}\n\n"
            "Please compile these into a cohesive 5-day itinerary."
        )

        await Console(travel_summary_agent.run_stream(task=summary_task))

    except Exception as e:
        print("Error while running agents:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
