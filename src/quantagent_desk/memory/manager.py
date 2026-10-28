"""Simple in-memory + optional SQLite session / entity / episodic memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MemoryManager:
    """
    Three layers:
    - short-term: conversation turns per session_id
    - entity: portfolio / counterparty facts
    - episodic: past stress runs, limit breaches, key calculations
    """

    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.episodes: List[Dict[str, Any]] = []

    # ---------- Short-term / session ----------
    def add_turn(self, session_id: str, role: str, content: str, meta: Optional[Dict] = None) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        })
        # keep last 20 turns
        self.sessions[session_id] = self.sessions[session_id][-20:]

    def get_history(self, session_id: str, last_n: int = 6) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])[-last_n:]

    def get_context_string(self, session_id: str, last_n: int = 2) -> str:
        from quantagent_desk.utils.tokens import clip_text

        turns = self.get_history(session_id, last_n)
        if not turns:
            return ""
        # Two short turns. Shock building and research do not replay the full desk log.
        lines = [f"{t['role'].upper()}: {clip_text(t['content'], 40)}" for t in turns]
        return clip_text("\n".join(lines), 100)

    # ---------- Entity memory ----------
    def upsert_entity(self, entity_id: str, data: Dict[str, Any]) -> None:
        if entity_id not in self.entities:
            self.entities[entity_id] = {"id": entity_id, "created": datetime.now(timezone.utc).isoformat()}
        self.entities[entity_id].update(data)
        self.entities[entity_id]["updated"] = datetime.now(timezone.utc).isoformat()

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)

    # ---------- Episodic ----------
    def add_episode(self, kind: str, summary: str, payload: Optional[Dict] = None) -> str:
        eid = str(uuid4())[:8]
        self.episodes.append({
            "id": eid,
            "kind": kind,  # "stress", "breach", "query", "calculation"
            "summary": summary,
            "payload": payload or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # keep last 100
        self.episodes = self.episodes[-100:]
        return eid

    def recent_episodes(self, kind: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        items = self.episodes
        if kind:
            items = [e for e in items if e["kind"] == kind]
        return items[-limit:]


# Global singleton (in production use proper DI / Redis / DB)
_memory = MemoryManager()


def get_memory() -> MemoryManager:
    return _memory
