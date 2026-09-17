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

MODEL = "openai/gpt-oss-120b"
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
5b. Before deciding, also call search_tickets to check the ticket queue for precedent - prior similar cases, how they were resolved or rejected, and their KB citations. Use this for consistency (e.g. an ungrounded access request should be treated the same way a similar past ticket was).
6. If a request has no KB coverage at all (e.g. a request for admin/server access, which no KB article grants), do not resolve it yourself - escalate it to a human with that reasoning. Check the ticket queue history for precedent (e.g. TK-1050 was rejected for exactly this kind of ungrounded access request) and mention it if relevant.
6b. Any suspected phishing, malware, or unauthorized-access report (KB-09) must use the escalate tool with assigned_to set to "Security" - never create_ticket - matching the TK-1048 precedent, even if you're also telling the employee not to forward it further.
7. End every request-handling turn with exactly ONE terminal action: create_ticket (for a fresh request you can resolve or that is already covered by clear, unconflicted policy), update_ticket (only when triaging a ticket that ALREADY EXISTS in the queue, identified by a TK-#### id given to you in the task - never invent a new ticket for it), escalate (for anything requiring human/Finance/Security/manager judgment, conflicts, or no-KB-coverage cases), or ask_clarifying_question (for vague requests). Do not call more than one terminal tool for the same request.
7b. When you are asked to triage an existing open ticket (the task will give you its TK-#### id and current status), do NOT call create_ticket - that would duplicate it. Call update_ticket with that same ticket_id to record your decision (e.g. confirm it's correctly still pending, move it forward, or change its status/action), or call escalate if it needs to be re-routed to a different or additional team.
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
                "required": ["employee", "issue_summary", "category", "action_taken", "kb_cited", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickets",
            "description": "Search the ticket queue (historical + agent-created) for precedent - prior similar cases and how they were resolved. Call this before deciding, alongside search_kb.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords describing the case, e.g. 'admin access request' or 'phishing'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_ticket",
            "description": "Update an existing ticket already in the queue (identified by its TK-#### id). Use this ONLY when triaging a ticket that already exists - never for a fresh employee request, which should use create_ticket instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The existing ticket's id, e.g. 'TK-1044'."},
                    "action_taken": {"type": "string"},
                    "status": {"type": "string"},
                    "kb_cited": {"type": "array", "items": {"type": "string"}},
                    "assigned_to": {"type": "string"},
                },
                "required": ["ticket_id", "action_taken", "status"],
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
                    "existing_ticket_id": {"type": "string", "description": "Set this ONLY if you are re-routing a ticket that already exists in the queue (e.g. 'TK-1044'), so it gets updated instead of duplicated. Omit for a fresh employee request."},
                },
                "required": ["employee", "issue_summary", "category", "reason", "assigned_to", "kb_cited"],
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


TERMINAL_TOOLS = {"create_ticket", "update_ticket", "escalate", "ask_clarifying_question"}


def _execute_tool(name: str, tool_input: dict) -> dict:
    if name == "search_kb":
        result = tools.search_kb(tool_input.get("query", ""))
    elif name == "search_tickets":
        result = tools.search_tickets(tool_input.get("query", ""))
    elif name == "create_ticket":
        result = tools.create_ticket(**tool_input)
    elif name == "update_ticket":
        result = tools.update_ticket(**tool_input)
    elif name == "escalate":
        action_taken = f"Escalated to {tool_input['assigned_to']}: {tool_input['reason']}"
        existing_id = tool_input.get("existing_ticket_id")
        if existing_id:
            result = tools.update_ticket(
                ticket_id=existing_id,
                action_taken=action_taken,
                status="Escalated",
                kb_cited=tool_input.get("kb_cited", []),
                assigned_to=tool_input["assigned_to"],
            )
        else:
            result = tools.create_ticket(
                employee=tool_input["employee"],
                email=tool_input.get("email"),
                issue_summary=tool_input["issue_summary"],
                category=tool_input["category"],
                action_taken=action_taken,
                kb_cited=tool_input.get("kb_cited", []),
                status="Escalated",
                assigned_to=tool_input["assigned_to"],
            )
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
    terminal_fired = False

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
            is_terminal = tc.function.name in TERMINAL_TOOLS
            if is_terminal and terminal_fired:
                # A terminal action already happened this turn; refuse a second
                # one instead of trusting the model to police rule #7 itself.
                result = {"error": "A terminal action was already taken for this request. Do not call another - just write your final reply to the employee now."}
            else:
                result = _execute_tool(tc.function.name, tool_input)
                if is_terminal:
                    terminal_fired = True
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


def handle_ticket_triage(ticket: dict, api_key: str = None):
    """Run an EXISTING open ticket from the queue through the same
    triage judgment as a fresh employee request (per the assignment's
    ticket-queue instructions), instead of only ever creating new tickets
    from chat messages."""
    message = (
        f"[EXISTING OPEN TICKET {ticket['id']}] Triage this ticket already in the queue - "
        f"do not create a new ticket for it.\n"
        f"Employee: {ticket['employee']}\n"
        f"Issue: {ticket['issue_summary']}\n"
        f"Category: {ticket.get('category', '')}\n"
        f"Current status: {ticket['status']}\n"
        f"Currently assigned to: {ticket.get('assigned_to') or 'unassigned'}\n"
        f"Prior action taken: {ticket.get('action_taken', '')}\n"
        f"Decide whether to update it (call update_ticket with ticket_id='{ticket['id']}') "
        f"or escalate/re-route it (call escalate with existing_ticket_id='{ticket['id']}')."
    )
    return handle_request(message, employee=ticket["employee"], email=ticket.get("email"), api_key=api_key)
