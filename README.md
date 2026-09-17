# Veridian Corp IT Support Agent

Internal service agent prototype for AIONOS Assignment 2. Given an employee
IT request, it searches the company knowledge base, decides whether to
resolve it, escalate it to a human team, or ask a clarifying question, and
logs every decision (with cited KB-IDs) to an audit trail.
# Deployed Link
https://it-support-agent-brj3tljkdu5snkm3g6bhky.streamlit.app/

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste a free Groq API key (console.groq.com) into it
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints. Pick a sample request (REQ-01..REQ-15)
from the sidebar or type your own, and watch the agent's tool calls,
resulting ticket, and audit trail in the other tabs.

## How it works

- `data/kb.json` — the 10 KB policy articles + the Finance asset policy extract.
- `data/tickets_seed.json` — the historical ticket queue (TK-1042..TK-1051), used as precedent.
- `data/requests.json` — the 15 sample employee requests for the demo.
- `app/tools.py` — plain-Python data layer (KB lookup, ticket read/write, audit log append).
- `app/agent.py` — the Claude/LLM tool-use loop. The model must call `search_kb` before deciding,
  then exactly one terminal tool: `create_ticket`, `escalate`, or `ask_clarifying_question`.
- `streamlit_app.py` — reviewer UI: Chat / Tickets / Audit Log / Knowledge Base tabs.

Runs on Groq (free tier, OpenAI-compatible tool calling) so no paid API key is required.
