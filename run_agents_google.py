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
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

async def main():
    # Try a few candidate Gemini model ids until one works (handles quota/restrictions)
    candidate_models = [
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "models/gemini-2.5-pro",
    ]

    # Provide minimal model_info so AutoGen knows these are Gemini models
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.GEMINI_2_5_FLASH,
        "structured_output": True,
        "multiple_system_messages": True,
    }

    model_client = None
    last_exc = None
    for candidate in candidate_models:
        try:
            print(f"Attempting model: {candidate}")
            model_client = OpenAIChatCompletionClient(
                model=candidate,
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model_info=model_info,
            )
            break
        except Exception as e:
            last_exc = e
            print(f"Model {candidate} failed to initialize: {e}")

    if model_client is None:
        print("Failed to create any model client:", last_exc)
        return

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

    travel_summary_agent = AssistantAgent(
        "travel_summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message=(
            "You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. "
            "You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE."
        ),
    )

    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[planner_agent, local_agent, language_agent, travel_summary_agent],
        termination_condition=termination,
    )

    print("Starting agents run (this may take a few seconds)...")
    try:
        await Console(team.run_stream(task="Plan a 5-day trip to Mexico City. Provide suggestions for activities, places to visit, and language tips."))
    except Exception as e:
        print("Error while running agents:", e)

if __name__ == "__main__":
    asyncio.run(main())
