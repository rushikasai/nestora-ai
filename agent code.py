import os
from datetime import datetime, timezone
from uuid import uuid4

from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents.experimental.chat_agent.protocol import (
    build_llm_message_history,
)
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# HomeWise AI — hosted on Agentverse
agent = Agent()
protocol = Protocol(spec=chat_protocol_spec)

SYSTEM_PROMPT = """
You are HomeWise AI, a helpful home-planning assistant.

Your job is to help users plan a practical concept layout for a family home.

Remember all plot and house requirements available in the supplied
conversation history. Never ask for details the user has already provided.

Collect these details when needed:
1. Plot width and length, including units.
2. City or location, if relevant to local climate and orientation.
3. Number of floors.
4. Family size and required bedrooms.
5. Other needs, such as kitchen, bathrooms, parking, stairs,
   balcony, garden, or a prayer room.
6. Preferred direction or entrance, if known.

CONVERSATION RULES:
- Ask only for the most important missing details.
- Ask no more than two related questions in one reply.
- If the user gives several details together, keep all of them in mind.
- If the user answers a previous question briefly, interpret it using
  the conversation history.
- Do not restart the interview after every message.
- When enough information is available, generate a concept plan.
- If some optional details are missing, make reasonable assumptions
  and clearly state them instead of repeatedly questioning the user.
- If the plot dimensions are missing, ask for them first.
- If the user says "change the kitchen", "add a bedroom", or similar,
  revise the existing plan using the known requirements.

PLAN OUTPUT:
1. Summarize the confirmed requirements.
2. Give a suggested floor-by-floor layout.
3. Provide a simple text diagram with room labels.
4. Explain the suggested room arrangement.
5. List assumptions and details that need confirmation.

Be friendly, clear, practical, and concise.
Use metric units where possible.

IMPORTANT:
- This is a conceptual planning assistant, not a licensed architect.
- Do not claim that a concept plan is structurally approved or ready
  for construction.
- Actual construction needs a qualified architect/engineer and
  compliance with local building rules.
- Never invent exact legal setbacks, structural dimensions, or approvals.
"""

# Agentverse Hosted Agents supply these environment variables.
client = OpenAI(
    api_key=os.environ["ASI1_API_KEY"],
    base_url=os.environ["ASI1_BASE_URL"],
)


def make_reply(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


@protocol.on_message(ChatMessage)
async def handle_message(
    ctx: Context,
    sender: str,
    msg: ChatMessage,
):
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    user_text = msg.text().strip()

    if not user_text:
        await ctx.send(
            sender,
            make_reply(
                "Tell me about your plot or the changes you want "
                "to make to your house plan."
            ),
        )
        return

    try:
        # Use the conversation history provided by Agentverse.
        history = build_llm_message_history(ctx)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
        ]

        # Avoid sending the latest user message twice if the helper
        # already includes it in the conversation history.
        if not messages or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model="asi1",
            messages=messages,
            max_tokens=1400,
        )

        answer = response.choices[0].message.content

        if not answer or not str(answer).strip():
            answer = (
                "I couldn't create a response this time. "
                "Please send your last message again."
            )

        await ctx.send(sender, make_reply(str(answer).strip()))

    except Exception:
        ctx.logger.exception("HomeWise AI request failed")
        await ctx.send(
            sender,
            make_reply(
                "Sorry, I couldn't process that request right now. "
                "Please try again. If the problem continues, check "
                "the Agentverse logs for the error."
            ),
        )


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(
    ctx: Context,
    sender: str,
    msg: ChatAcknowledgement,
):
    pass


agent.include(protocol, publish_manifest=True)

# Do not add agent.run() for an Agentverse-hosted agent.
