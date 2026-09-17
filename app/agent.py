"""Core agent loop: LLM + tool use over the IT-support KB/ticket data.

Runs on Groq (free tier, OpenAI-compatible tool-calling API) so the
prototype needs no paid API key. The model is the reasoning core: it
decides, per employee message, whether to look up policy, create a ticket,
escalate to a human team, or ask a clarifying question - then produces a
final reply to the employee. Every tool call it makes is written to the
audit log as it happens, so the audit trail doesn't depend on the model
remembering to log anything itself.
"""
import json
import os
import time

from groq import Groq, RateLimitError

from app import tools

MODEL = "openai/gpt-oss-20b"
MAX_TOOL_ITERATIONS = 6

SYSTEM_PROMPT = """You are the internal IT Support agent for Veridian Corp.

GROUND RULES (non-negotiable):
1. Use ONLY the knowledge base returned by search_kb as your source of policy truth. Never invent a policy.
2. Every resolution you give must cite the specific KB-ID(s) it relied on, OR explicitly state that no policy in the KB covers the case.
3. Always call search_kb at least once before deciding how to handle a request, even if you think you know the answer.
4. Watch for TWO kinds of traps in employee messages, not just the surface-level ask:
   - Conflicting policies: if two KB/policy articles disagree (e.g. a KB article's eligibility window vs. the Asset Management Policy's refresh cycle), surface BOTH to the employee/human reviewer and do not silently pick one. Use escalate or create_ticket with a status that makes the conflict visible - never auto-approve across a conflict.
   - Embedded violations: a request can be mostly fine but contain a separate policy violation inside it (e.g. someone reporting a phishing email but says they also forwarded it to teammates, which itself violates KB-09). Call out the embedded violation explicitly in your reply even while handling the primary ask.
5. If a request is too vague to act on (e.g. "it's not working" with no detail), do NOT guess. Use ask_clarifying_question instead of creating a ticket.
6. If a request has no KB coverage at all (e.g. a request for admin/server access, which no KB article grants), do not resolve it yourself - escalate it to a human with that reasoning. Check the ticket queue history for precedent (e.g. TK-1050 was rejected for exactly this kind of ungrounded access request) and mention it if relevant.
7. End every request-handling turn with exactly ONE terminal action: create_ticket (for things you can resolve or that are already covered by clear, unconflicted policy), escalate (for anything requiring human/Finance/Security/manager judgment, conflicts, or no-KB-coverage cases), or ask_clarifying_question (for vague requests). Do not call more than one terminal tool for the same request.
8. After the terminal tool call, ALWAYS write a short final reply as plain, natural-language prose addressed directly to the employee. Never output raw JSON, a tool call, or an echo of a tool's arguments/result as your final reply - not even for ask_clarifying_question. If the terminal action was ask_clarifying_question, your final reply IS the question, phrased conversationally (e.g. "Could you tell me more about..." rather than {"question": "..."}).

Categories to classify every request into (for your own reasoning): auto-resolve, resolve-with-info, ask-clarifying-question, escalate.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the IT policy knowledge base. Returns matching KB articles with their id, title, and text. Always call this before deciding how to handle a request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords describing the topic, e.g. 'laptop replacement' or 'phishing'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a ticket for a request you can resolve directly or that is already covered by clear, unconflicted policy. Do not use this for cases needing human judgment - use escalate instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee": {"type": "string"},
                    "email": {"type": "string"},
                    "issue_summary": {"type": "string"},
                    "category": {"type": "string"},
                    "action_taken": {"type": "string", "description": "What was done or what the employee should do next."},
                    "kb_cited": {"type": "array", "items": {"type": "string"}, "description": "KB-IDs relied on, e.g. ['KB-01']."},
                    "status": {"type": "string", "description": "e.g. Resolved, Approved, In Progress."},
                },
                "required": ["employee", "email", "issue_summary", "category", "action_taken", "kb_cited", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate a request to a human team (IT Security, Finance, a manager, etc.) because it requires human judgment, involves a policy conflict, has no KB coverage, or involves a security incident.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee": {"type": "string"},
                    "email": {"type": "string"},
                    "issue_summary": {"type": "string"},
                    "category": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this needs a human, e.g. 'conflicting policies' or 'no KB coverage for admin access, see TK-1050 precedent'."},
                    "assigned_to": {"type": "string", "description": "e.g. IT Security, Finance, Employee's Manager."},
                    "kb_cited": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["employee", "email", "issue_summary", "category", "reason", "assigned_to", "kb_cited"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarifying_question",
            "description": "Use when the request is too vague to act on. Does not create a ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"],
            },
        },
    },
]


def _create_with_retry(client, messages, retries=4):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                max_tokens=1024,
                tools=TOOLS,
                tool_choice="auto",
                messages=messages,
            )
        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _execute_tool(name: str, tool_input: dict) -> dict:
    if name == "search_kb":
        result = tools.search_kb(tool_input.get("query", ""))
    elif name == "create_ticket":
        result = tools.create_ticket(**tool_input)
    elif name == "escalate":
        ticket_input = {
            "employee": tool_input["employee"],
            "email": tool_input["email"],
            "issue_summary": tool_input["issue_summary"],
            "category": tool_input["category"],
            "action_taken": f"Escalated to {tool_input['assigned_to']}: {tool_input['reason']}",
            "kb_cited": tool_input.get("kb_cited", []),
            "status": "Escalated",
            "assigned_to": tool_input["assigned_to"],
        }
        result = tools.create_ticket(**ticket_input)
    elif name == "ask_clarifying_question":
        result = {"question": tool_input["question"]}
    else:
        result = {"error": f"unknown tool {name}"}

    tools.append_audit({
        "tool": name,
        "input": tool_input,
        "result": result,
    })
    return result


def handle_request(employee_message: str, employee: str = None, email: str = None,
                    api_key: str = None, messages: list = None):
    """Run one employee message through the agent loop.

    `messages` lets a caller continue an existing thread (e.g. after the
    agent asked a clarifying question): pass back the `messages` list
    returned from the previous call. Returns a dict with the final reply
    text, the list of tool calls made, and the updated `messages` list to
    persist for the next turn."""
    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    context_lines = [employee_message]
    if employee and messages is None:
        context_lines.insert(0, f"Employee: {employee} <{email}>")
    user_content = "\n".join(context_lines)

    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages = list(messages) + [{"role": "user", "content": user_content}]
    tool_calls_made = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = _create_with_retry(client, messages)
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            return {"reply": message.content or "", "tool_calls": tool_calls_made, "messages": messages}

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            tool_input = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, tool_input)
            tool_calls_made.append({"name": tc.function.name, "input": tool_input, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return {
        "reply": "(Agent did not reach a final answer within the tool-call budget.)",
        "tool_calls": tool_calls_made,
        "messages": messages,
    }
