"""Data-layer tool implementations backing the agent's tool calls.

Everything here is plain JSON-file I/O by design: the KB is 10 records,
tickets are a handful more, so a database would be overhead without
adding value for this prototype.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_PATH = DATA_DIR / "kb.json"
TICKETS_SEED_PATH = DATA_DIR / "tickets_seed.json"
TICKETS_PATH = DATA_DIR / "tickets.json"
AUDIT_PATH = DATA_DIR / "audit_log.json"

_lock = Lock()


def _load_json(path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_kb():
    return _load_json(KB_PATH, [])


def load_requests():
    return _load_json(DATA_DIR / "requests.json", [])


def reset_tickets():
    """(Re)seed the runtime ticket store from the historical ticket queue."""
    seed = _load_json(TICKETS_SEED_PATH, [])
    _save_json(TICKETS_PATH, seed)


def reset_audit_log():
    _save_json(AUDIT_PATH, [])


def load_tickets():
    if not TICKETS_PATH.exists():
        reset_tickets()
    return _load_json(TICKETS_PATH, [])


def load_audit_log():
    if not AUDIT_PATH.exists():
        reset_audit_log()
    return _load_json(AUDIT_PATH, [])


def _next_ticket_id():
    tickets = load_tickets()
    max_num = 1051
    for t in tickets:
        m = re.match(r"TK-(\d+)", t["id"])
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"TK-{max_num + 1}"


def search_kb(query: str = ""):
    """Return the full KB. The corpus is only 10 articles, so there is no
    retrieval-quality problem to solve here - the model reasons over all of
    them directly rather than relying on lexical/semantic matching to narrow
    them down first. `query` is accepted (and logged) for audit purposes but
    does not filter results. Revisit with real retrieval (embeddings) only if
    the KB grows large enough that dumping it whole stops being viable.
    """
    return load_kb()


def create_ticket(employee: str, email: str, issue_summary: str, category: str,
                   action_taken: str, kb_cited: list, status: str,
                   assigned_to: str = None):
    with _lock:
        tickets = load_tickets()
        ticket = {
            "id": _next_ticket_id(),
            "employee": employee,
            "email": email,
            "issue_summary": issue_summary,
            "category": category,
            "kb_cited": kb_cited or [],
            "action_taken": action_taken,
            "status": status,
            "assigned_to": assigned_to,
            "created_at": datetime.now().isoformat(),
            "source": "agent",
        }
        tickets.append(ticket)
        _save_json(TICKETS_PATH, tickets)
        return ticket


def append_audit(entry: dict):
    with _lock:
        log = load_audit_log()
        entry = {"timestamp": datetime.now().isoformat(), **entry}
        log.append(entry)
        _save_json(AUDIT_PATH, log)
        return entry
