"""Reviewer-facing UI for the Veridian Corp IT Support agent.

Run with: streamlit run streamlit_app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

from app import agent, tools

load_dotenv()

st.set_page_config(page_title="Veridian IT Support Agent", layout="wide")

if "thread" not in st.session_state:
    st.session_state.thread = None  # raw Groq messages list, for continuation
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [{role, content, tool_calls?}]
if "current_employee" not in st.session_state:
    st.session_state.current_employee = {"employee": "", "email": ""}

with st.sidebar:
    st.header("Veridian IT Support Agent")
    api_key = st.text_input(
        "Groq API key", value=os.environ.get("GROQ_API_KEY", ""), type="password",
        help="Free key from console.groq.com. Falls back to GROQ_API_KEY env var / .env file.",
    )

    st.subheader("Load a sample request")
    requests = tools.load_requests()
    req_labels = {r["id"]: f"{r['id']} - {r['employee']}: {r['request'][:40]}..." for r in requests}
    chosen_id = st.selectbox("Employee requests (REQ-01..15)", options=[""] + list(req_labels), format_func=lambda k: req_labels.get(k, "-- pick one --"))
    load_clicked = st.button("Load into chat", use_container_width=True)

    st.divider()
    st.subheader("Triage an active ticket")
    st.caption("Open cases already in the queue - the agent must resolve or re-route these too, not just fresh requests.")
    active_tickets = [t for t in tools.load_tickets() if t.get("active")]
    if active_tickets:
        tk_labels = {t["id"]: f"{t['id']} - {t['employee']}: {t['issue_summary'][:35]}..." for t in active_tickets}
        chosen_tk = st.selectbox("Active tickets", options=[""] + list(tk_labels), format_func=lambda k: tk_labels.get(k, "-- pick one --"))
        triage_clicked = st.button("Triage this ticket", use_container_width=True)
    else:
        chosen_tk, triage_clicked = None, False
        st.caption("No active tickets left to triage.")

    st.divider()
    if st.button("Reset demo data (tickets + audit log)", use_container_width=True):
        tools.reset_tickets()
        tools.reset_audit_log()
        st.session_state.thread = None
        st.session_state.chat_display = []
        st.success("Reset.")

    if st.button("New conversation", use_container_width=True):
        st.session_state.thread = None
        st.session_state.chat_display = []
        st.session_state.current_employee = {"employee": "", "email": ""}

    st.divider()
    reviewer_mode = st.toggle(
        "Reviewer mode", value=True,
        help="Shows Tickets/Audit Log/Knowledge Base tabs and the agent's tool calls. "
             "Turn off to see only what an employee would see.",
    )

if load_clicked and chosen_id:
    req = next(r for r in requests if r["id"] == chosen_id)
    st.session_state.thread = None
    st.session_state.chat_display = []
    st.session_state.current_employee = {"employee": req["employee"], "email": req["email"]}
    st.session_state.pending_message = req["request"]

if triage_clicked and chosen_tk:
    ticket = next(t for t in active_tickets if t["id"] == chosen_tk)
    st.session_state.thread = None
    st.session_state.chat_display = []
    st.session_state.current_employee = {"employee": ticket["employee"], "email": ticket.get("email") or ""}
    st.session_state.pending_ticket = ticket

def render_chat():
    emp = st.session_state.current_employee
    col1, col2 = st.columns(2)
    with col1:
        emp["employee"] = st.text_input("Employee name", value=emp["employee"])
    with col2:
        emp["email"] = st.text_input("Employee email", value=emp["email"])

    for turn in st.session_state.chat_display:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if reviewer_mode and turn.get("tool_calls"):
                with st.expander("Agent actions (tool calls)"):
                    for tc in turn["tool_calls"]:
                        st.markdown(f"**{tc['name']}**")
                        st.json({"input": tc["input"], "result": tc["result"]})


if reviewer_mode:
    tab_chat, tab_tickets, tab_audit, tab_kb = st.tabs(["Chat", "Tickets", "Audit Log", "Knowledge Base"])

    with tab_chat:
        render_chat()

    with tab_tickets:
        st.subheader("Ticket queue")
        st.dataframe(tools.load_tickets(), use_container_width=True)

    with tab_audit:
        st.subheader("Audit trail")
        st.json(tools.load_audit_log())

    with tab_kb:
        st.subheader("Knowledge base")
        for article in tools.load_kb():
            with st.expander(f"{article['id']} - {article['title']}"):
                st.write(article["text"])
else:
    render_chat()

# st.chat_input can't live inside st.tabs/columns/sidebar, so it's placed
# here at the page level; it still visually pins to the bottom of the page.
pending_ticket = st.session_state.pop("pending_ticket", None)
pending = st.session_state.pop("pending_message", None)
user_msg = st.chat_input("Type an employee request...") or pending

if pending_ticket:
    if not api_key:
        st.error("Enter a Groq API key in the sidebar first.")
    else:
        st.session_state.chat_display.append({
            "role": "user",
            "content": f"[Triage {pending_ticket['id']}] {pending_ticket['issue_summary']} (current status: {pending_ticket['status']}, assigned to: {pending_ticket.get('assigned_to') or 'unassigned'})",
        })
        with st.spinner("Agent triaging ticket..."):
            result = agent.handle_ticket_triage(pending_ticket, api_key=api_key)
        st.session_state.thread = result["messages"]
        st.session_state.chat_display.append({
            "role": "assistant",
            "content": result["reply"],
            "tool_calls": result["tool_calls"],
        })
        st.rerun()
elif user_msg:
    if not api_key:
        st.error("Enter a Groq API key in the sidebar first.")
    else:
        emp = st.session_state.current_employee
        st.session_state.chat_display.append({"role": "user", "content": user_msg})
        with st.spinner("Agent thinking..."):
            result = agent.handle_request(
                user_msg,
                employee=emp["employee"] or None,
                email=emp["email"] or None,
                api_key=api_key,
                messages=st.session_state.thread,
            )
        st.session_state.thread = result["messages"]
        st.session_state.chat_display.append({
            "role": "assistant",
            "content": result["reply"],
            "tool_calls": result["tool_calls"],
        })
        st.rerun()
