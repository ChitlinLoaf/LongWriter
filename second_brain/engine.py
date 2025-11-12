"""Core implementation of the Gemini-inspired second brain pipeline.

This module provides a light-weight cognitive orchestration engine capable of
consuming reminder and calendar data, normalising the information, and producing
behavioural feedback loops.  The implementation is intentionally heuristic so it
can operate on locally available data without external API access while still
providing a deterministic, inspectable pipeline that mirrors the high-level
requirements in the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import re
import uuid


def _ensure_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 strings while gracefully handling missing data."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+0000"
        # allow timezone without colon
        if re.match(r".*[+-]\d{2}:?\d{2}$", value):
            value = re.sub(r"([+-]\d{2})(:)?(\d{2})$", lambda m: f"{m.group(1)}{m.group(3)}", value)
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


def _default(obj: Any) -> Any:
    """Default JSON serialiser for datetime aware objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


@dataclass
class Reminder:
    identifier: str
    title: str
    notes: str = ""
    due_date: Optional[datetime] = None
    creation_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    completed: bool = False
    recurrence: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CalendarEvent:
    identifier: str
    title: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    notes: str = ""
    location: Optional[str] = None
    all_day: bool = False
    recurrence: Optional[str] = None
    attendees: List[str] = field(default_factory=list)


@dataclass
class CognitiveSignal:
    identifier: str
    source: str  # "reminder" or "event"
    title: str
    notes: str
    start: Optional[datetime]
    end: Optional[datetime]
    due: Optional[datetime]
    completed: bool
    recurrence: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class TemporalEdge:
    origin: str
    target: str
    weight: float
    relation: str


@dataclass
class BehaviouralPrompt:
    label: str
    cadence: str
    trigger: str
    payload: Dict[str, Any]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class SignalIngestor:
    DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "⛑ Health": ("health", "fitness", "exercise", "doctor", "med", "sleep"),
        "⚖️ Admin": ("invoice", "paperwork", "schedule", "renew", "license", "form"),
        "🧬 Creative": ("write", "design", "compose", "draft", "brainstorm", "paint"),
        "📣 Advocacy": ("campaign", "speak", "meeting", "organize", "advocacy", "outreach"),
        "🧠 Learning": ("study", "learn", "course", "reading", "research", "tutorial"),
        "💰 Finance": ("budget", "tax", "invoice", "payment", "expense", "salary"),
    }

    ENERGY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "low": ("email", "call", "text", "review", "scan", "reply"),
        "medium": ("prepare", "write", "draft", "plan", "design"),
        "high": ("presentation", "research", "workout", "deploy", "analysis"),
    }

    def __init__(self, now: Optional[datetime] = None) -> None:
        self.now = now or datetime.now().astimezone()

    def process(self, reminders: Iterable[Reminder], events: Iterable[CalendarEvent]) -> List[CognitiveSignal]:
        signals: List[CognitiveSignal] = []
        for reminder in reminders:
            metadata = self._label(reminder.title, reminder.notes, reminder.tags, reminder.due_date, reminder.recurrence)
            signals.append(
                CognitiveSignal(
                    identifier=reminder.identifier or str(uuid.uuid4()),
                    source="reminder",
                    title=reminder.title.strip(),
                    notes=reminder.notes.strip(),
                    start=None,
                    end=reminder.due_date,
                    due=reminder.due_date,
                    completed=reminder.completed,
                    recurrence=reminder.recurrence,
                    metadata=metadata,
                )
            )
        for event in events:
            metadata = self._label(event.title, event.notes, event.attendees, event.start_date, event.recurrence)
            signals.append(
                CognitiveSignal(
                    identifier=event.identifier or str(uuid.uuid4()),
                    source="event",
                    title=event.title.strip(),
                    notes=event.notes.strip(),
                    start=event.start_date,
                    end=event.end_date,
                    due=event.start_date,
                    completed=False,
                    recurrence=event.recurrence,
                    metadata=metadata,
                )
            )
        return signals

    def _label(
        self,
        title: str,
        notes: str,
        tags: Iterable[str],
        due: Optional[datetime],
        recurrence: Optional[str],
    ) -> Dict[str, Any]:
        text = f"{title} {notes} {' '.join(tags)}".lower()
        domain = self._classify_domain(text)
        urgency = self._calculate_urgency(due)
        energy = self._estimate_energy(text)
        recurrence_type = self._describe_recurrence(recurrence)
        drift = self._estimate_drift(due, recurrence_type)
        return {
            "domain": domain,
            "urgency": urgency,
            "energy_cost": energy,
            "recurrence_type": recurrence_type,
            "drift_potential": drift,
        }

    def _classify_domain(self, text: str) -> str:
        for label, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return label
        return "⚖️ Admin"  # default administrative bucket

    def _calculate_urgency(self, due: Optional[datetime]) -> str:
        if not due:
            return "unscheduled"
        delta = (due - self.now).total_seconds() / 3600
        if delta <= 0:
            return "overdue"
        if delta < 24:
            return "imminent"
        if delta < 72:
            return "soon"
        return "scheduled"

    def _estimate_energy(self, text: str) -> str:
        for energy, keywords in self.ENERGY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return energy
        return "medium"

    def _describe_recurrence(self, recurrence: Optional[str]) -> str:
        if not recurrence:
            return "single"
        recurrence = recurrence.lower()
        if "week" in recurrence:
            return "weekly"
        if "month" in recurrence:
            return "monthly"
        if "day" in recurrence:
            return "daily"
        return "custom"

    def _estimate_drift(self, due: Optional[datetime], recurrence_type: str) -> float:
        if not due:
            return 0.2 if recurrence_type == "single" else 0.4
        delta = (self.now - due).total_seconds() / 86400
        if delta <= 0:
            return 0.1
        base = min(0.9, max(0.2, delta / 14))
        if recurrence_type != "single":
            base *= 0.8
        return round(base, 2)


class TemporalGraphEngine:
    def build(self, signals: List[CognitiveSignal]) -> Dict[str, Any]:
        edges: List[TemporalEdge] = []
        idle_loops: List[str] = []
        chains: Dict[str, List[str]] = {}

        title_index = {signal.identifier: self._tokenise(signal.title) for signal in signals}
        domain_index: Dict[str, List[str]] = {}
        for signal in signals:
            domain_index.setdefault(signal.metadata["domain"], []).append(signal.identifier)

        for signal in signals:
            similar = self._find_similar(signals, signal, title_index)
            for target_id, weight in similar:
                edges.append(TemporalEdge(origin=signal.identifier, target=target_id, weight=weight, relation="semantic"))

            sequential = self._link_sequential(signal, signals)
            for target_id in sequential:
                edges.append(TemporalEdge(origin=signal.identifier, target=target_id, weight=0.6, relation="temporal"))

            if signal.recurrence and signal.metadata["drift_potential"] > 0.5:
                idle_loops.append(signal.identifier)

        for domain, ids in domain_index.items():
            if len(ids) > 1:
                chain_id = f"chain::{_slugify(domain)}"
                chains[chain_id] = sorted(ids)

        bottlenecks = self._detect_bottlenecks(signals, edges)
        bursts = self._find_unanchored_bursts(signals)

        return {
            "nodes": [self._serialise_signal(signal) for signal in signals],
            "edges": [edge.__dict__ for edge in edges],
            "idle_loops": idle_loops,
            "bottlenecks": bottlenecks,
            "unanchored_bursts": bursts,
            "chains": chains,
        }

    def _tokenise(self, text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _find_similar(
        self,
        signals: List[CognitiveSignal],
        anchor: CognitiveSignal,
        title_index: Dict[str, set],
    ) -> List[Tuple[str, float]]:
        anchor_tokens = title_index[anchor.identifier]
        scores: List[Tuple[str, float]] = []
        for signal in signals:
            if signal.identifier == anchor.identifier:
                continue
            tokens = title_index[signal.identifier]
            if not tokens:
                continue
            overlap = len(anchor_tokens & tokens)
            if overlap == 0:
                continue
            total = len(anchor_tokens | tokens)
            weight = round(overlap / total, 2)
            if weight >= 0.2:
                scores.append((signal.identifier, weight))
        return scores

    def _link_sequential(self, anchor: CognitiveSignal, signals: List[CognitiveSignal]) -> List[str]:
        if not anchor.due:
            return []
        results: List[str] = []
        for signal in signals:
            if signal.identifier == anchor.identifier or not signal.due:
                continue
            delta = (signal.due - anchor.due).total_seconds() / 3600
            if 0 < delta <= 72 and signal.metadata["domain"] == anchor.metadata["domain"]:
                results.append(signal.identifier)
        return results

    def _detect_bottlenecks(self, signals: List[CognitiveSignal], edges: List[TemporalEdge]) -> List[str]:
        incoming: Dict[str, int] = {}
        outgoing: Dict[str, int] = {}
        for edge in edges:
            outgoing[edge.origin] = outgoing.get(edge.origin, 0) + 1
            incoming[edge.target] = incoming.get(edge.target, 0) + 1
        bottlenecks = []
        for signal in signals:
            if outgoing.get(signal.identifier, 0) >= 3 and incoming.get(signal.identifier, 0) == 0:
                bottlenecks.append(signal.identifier)
        return bottlenecks

    def _find_unanchored_bursts(self, signals: List[CognitiveSignal]) -> List[str]:
        bursts = []
        for signal in signals:
            if signal.metadata["urgency"] == "unscheduled" and not signal.due and signal.metadata["drift_potential"] >= 0.4:
                bursts.append(signal.identifier)
        return bursts

    def _serialise_signal(self, signal: CognitiveSignal) -> Dict[str, Any]:
        return {
            "identifier": signal.identifier,
            "source": signal.source,
            "title": signal.title,
            "notes": signal.notes,
            "start": signal.start,
            "end": signal.end,
            "due": signal.due,
            "completed": signal.completed,
            "recurrence": signal.recurrence,
            "metadata": signal.metadata,
        }


class StructuralOptimizer:
    def optimise(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph["nodes"]
        deduped_nodes, duplicates = self._deduplicate(nodes)
        canonical_nodes = [self._canonicalise(node) for node in deduped_nodes]
        chains = self._build_chains(graph, canonical_nodes)
        return {
            "nodes": canonical_nodes,
            "duplicates": duplicates,
            "chains": chains,
        }

    def _deduplicate(self, nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        seen: Dict[str, str] = {}
        deduped: List[Dict[str, Any]] = []
        duplicates: List[Dict[str, Any]] = []
        for node in nodes:
            key = _slugify(node["title"])
            if key in seen:
                duplicates.append({"original": seen[key], "duplicate": node["identifier"]})
                continue
            seen[key] = node["identifier"]
            deduped.append(node)
        return deduped, duplicates

    def _canonicalise(self, node: Dict[str, Any]) -> Dict[str, Any]:
        canonical = dict(node)
        canonical["title"] = self._normalise_title(node["title"])
        if node.get("due") and isinstance(node["due"], datetime):
            canonical["due"] = node["due"].replace(microsecond=0)
        if node.get("start") and isinstance(node["start"], datetime):
            canonical["start"] = node["start"].replace(microsecond=0)
        if node.get("end") and isinstance(node["end"], datetime):
            canonical["end"] = node["end"].replace(microsecond=0)
        return canonical

    def _normalise_title(self, title: str) -> str:
        title = re.sub(r"\s+", " ", title).strip()
        if not title:
            return "Untitled"
        title = title[0].upper() + title[1:]
        return title

    def _build_chains(self, graph: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node_ids = {node["identifier"] for node in nodes}
        adjacency: Dict[str, List[str]] = {}
        for edge in graph["edges"]:
            if edge["origin"] in node_ids and edge["target"] in node_ids:
                adjacency.setdefault(edge["origin"], []).append(edge["target"])
        chains: List[Dict[str, Any]] = []
        for origin, targets in adjacency.items():
            if len(targets) >= 1:
                chains.append({"origin": origin, "targets": sorted(targets)})
        return chains


class BehaviouralLoopEngine:
    def generate(self, graph: Dict[str, Any], structures: Dict[str, Any], now: Optional[datetime] = None) -> List[BehaviouralPrompt]:
        now = now or datetime.now().astimezone()
        prompts: List[BehaviouralPrompt] = []
        prompts.append(
            BehaviouralPrompt(
                label="🧠 Sync Delta",
                cadence="weekly",
                trigger=(now + timedelta(days=(6 - now.weekday()))).replace(hour=8, minute=0, second=0, microsecond=0).isoformat(),
                payload={"focus": "Review new signals and confirm alignment."},
            )
        )
        lagging = [node for node in structures["nodes"] if node["metadata"].get("drift_potential", 0) > 0.6]
        if lagging:
            prompts.append(
                BehaviouralPrompt(
                    label="🕵️ Drift Checkpoint",
                    cadence=">72h",
                    trigger="lagging_items",
                    payload={"count": len(lagging), "items": [node["identifier"] for node in lagging]},
                )
            )
        priority_shifts = self._estimate_priority_shifts(graph)
        prompts.append(
            BehaviouralPrompt(
                label="🎯 Focus Realignment",
                cadence="dynamic",
                trigger=f"every {max(1, priority_shifts)} shifts",
                payload={"shifts_detected": priority_shifts},
            )
        )
        return prompts

    def _estimate_priority_shifts(self, graph: Dict[str, Any]) -> int:
        urgent_nodes = [node for node in graph["nodes"] if node["metadata"].get("urgency") in {"overdue", "imminent"}]
        return max(1, len(urgent_nodes) // 2)


class SnapshotEngine:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def capture(
        self,
        graph: Dict[str, Any],
        structures: Dict[str, Any],
        prompts: List[BehaviouralPrompt],
    ) -> Dict[str, Any]:
        previous = self._load_latest_snapshot("snapshot")
        current_state = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "structures": structures,
            "prompts": [prompt.__dict__ for prompt in prompts],
        }
        delta = self._diff(previous, current_state)
        self._append_behaviour_log(prompts)
        return {"current": current_state, "delta": delta}

    def _diff(self, previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, Any]:
        if not previous:
            return {"summary": "Initial snapshot", "changes": {}}
        prev_nodes = {node["identifier"]: node for node in previous.get("nodes", [])}
        curr_nodes = {node["identifier"]: node for node in current.get("nodes", [])}
        added = [curr_nodes[node_id] for node_id in curr_nodes.keys() - prev_nodes.keys()]
        removed = [prev_nodes[node_id] for node_id in prev_nodes.keys() - curr_nodes.keys()]
        updated = []
        for node_id in curr_nodes.keys() & prev_nodes.keys():
            prev = prev_nodes[node_id]
            curr = curr_nodes[node_id]
            if prev.get("metadata") != curr.get("metadata") or prev.get("due") != curr.get("due"):
                updated.append({"identifier": node_id, "from": prev.get("metadata"), "to": curr.get("metadata")})
        return {
            "summary": "Delta computed",
            "changes": {"added": added, "removed": removed, "updated": updated},
        }

    def _load_latest_snapshot(self, prefix: str) -> Optional[Dict[str, Any]]:
        snapshot_dir = self.base_path / "exports"
        if not snapshot_dir.exists():
            return None
        candidates = sorted(snapshot_dir.glob(f"{prefix}_*.json"))
        if not candidates:
            return None
        latest = candidates[-1]
        try:
            with latest.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return None

    def _append_behaviour_log(self, prompts: List[BehaviouralPrompt]) -> None:
        behaviours_path = self.base_path / "memory_bank" / "ea_behaviors.jsonl"
        behaviours_path.parent.mkdir(parents=True, exist_ok=True)
        with behaviours_path.open("a", encoding="utf-8") as handle:
            for prompt in prompts:
                handle.write(json.dumps(prompt.__dict__, ensure_ascii=False, default=_default) + "\n")


class StorageManager:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_raw(self, signals: List[CognitiveSignal]) -> Path:
        path = self._build_path("memory_bank", "raw")
        self._dump(path, [signal.__dict__ for signal in signals])
        return path

    def write_graph(self, graph: Dict[str, Any]) -> Path:
        path = self._build_path("memory_bank", "ea_temporal_graph")
        self._dump(path, graph)
        return path

    def write_structures(self, structures: Dict[str, Any]) -> Path:
        path = self._build_path("memory_bank", "normalized_structures")
        self._dump(path, structures)
        return path

    def write_snapshot(self, snapshot: Dict[str, Any]) -> Path:
        path = self._build_path("exports", "snapshot")
        self._dump(path, snapshot["current"])
        return path

    def _build_path(self, folder: str, prefix: str) -> Path:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        directory = self.base_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{prefix}_{timestamp}.json"

    def _dump(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=_default, indent=2)


class QueryMindInterface:
    def __init__(self, graph: Dict[str, Any], structures: Dict[str, Any]) -> None:
        self.graph = graph
        self.structures = structures

    def query(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()
        if "blocking" in question_lower:
            return self._find_blockers()
        if "decayed" in question_lower:
            return self._find_decay()
        if "fragmented" in question_lower:
            return self._attention_fragmentation()
        return {"response": "Query not recognised", "question": question}

    def _find_blockers(self) -> Dict[str, Any]:
        incoming: Dict[str, int] = {}
        for edge in self.graph["edges"]:
            incoming[edge["target"]] = incoming.get(edge["target"], 0) + 1
        blockers = [node for node in self.graph["nodes"] if incoming.get(node["identifier"], 0) >= 2]
        return {"blocking_nodes": blockers, "count": len(blockers)}

    def _find_decay(self) -> Dict[str, Any]:
        decay = [
            node
            for node in self.structures["nodes"]
            if node["metadata"].get("drift_potential", 0) >= 0.7 and node["metadata"].get("urgency") == "unscheduled"
        ]
        return {"decayed_goal_chains": decay, "count": len(decay)}

    def _attention_fragmentation(self) -> Dict[str, Any]:
        domains: Dict[str, int] = {}
        for node in self.graph["nodes"]:
            domain = node["metadata"].get("domain")
            domains[domain] = domains.get(domain, 0) + 1
        entropy = 0.0
        total = sum(domains.values()) or 1
        for count in domains.values():
            probability = count / total
            entropy -= probability * math.log(probability or 1, 2)
        return {"domain_distribution": domains, "attention_entropy": round(entropy, 2)}


class GeminiSecondBrain:
    def __init__(self, base_path: Optional[Path] = None) -> None:
        self.base_path = base_path or Path("second_brain")
        self.storage = StorageManager(self.base_path)
        self.snapshot_engine = SnapshotEngine(self.base_path)

    def run(
        self,
        reminders: Iterable[Dict[str, Any]],
        events: Iterable[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        parsed_reminders = [self._parse_reminder(item) for item in reminders]
        parsed_events = [self._parse_event(item) for item in events]

        ingestor = SignalIngestor(now=now)
        signals = ingestor.process(parsed_reminders, parsed_events)
        raw_path = self.storage.write_raw(signals)

        graph_engine = TemporalGraphEngine()
        graph = graph_engine.build(signals)
        graph_path = self.storage.write_graph(graph)

        optimizer = StructuralOptimizer()
        structures = optimizer.optimise(graph)
        structure_path = self.storage.write_structures(structures)

        behavioural_engine = BehaviouralLoopEngine()
        prompts = behavioural_engine.generate(graph, structures, now=now)

        snapshot = self.snapshot_engine.capture(graph, structures, prompts)
        snapshot_path = self.storage.write_snapshot(snapshot)

        query_interface = QueryMindInterface(graph, structures)

        return {
            "signals": signals,
            "graph": graph,
            "structures": structures,
            "prompts": [prompt.__dict__ for prompt in prompts],
            "snapshot": snapshot,
            "paths": {
                "raw": str(raw_path),
                "graph": str(graph_path),
                "structures": str(structure_path),
                "snapshot": str(snapshot_path),
                "behaviours": str(self.base_path / "memory_bank" / "ea_behaviors.jsonl"),
            },
            "query": query_interface,
        }

    def _parse_reminder(self, payload: Dict[str, Any]) -> Reminder:
        return Reminder(
            identifier=payload.get("id", str(uuid.uuid4())),
            title=payload.get("title", "Untitled"),
            notes=payload.get("notes", ""),
            due_date=_ensure_datetime(payload.get("due")),
            creation_date=_ensure_datetime(payload.get("created_at")),
            completion_date=_ensure_datetime(payload.get("completed_at")),
            completed=payload.get("completed", False),
            recurrence=payload.get("recurrence"),
            tags=payload.get("tags", []),
        )

    def _parse_event(self, payload: Dict[str, Any]) -> CalendarEvent:
        return CalendarEvent(
            identifier=payload.get("id", str(uuid.uuid4())),
            title=payload.get("title", "Untitled"),
            start_date=_ensure_datetime(payload.get("start")),
            end_date=_ensure_datetime(payload.get("end")),
            notes=payload.get("notes", ""),
            location=payload.get("location"),
            all_day=payload.get("all_day", False),
            recurrence=payload.get("recurrence"),
            attendees=payload.get("attendees", []),
        )


def demo_payload() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    now = datetime.now().astimezone()
    reminders = [
        {
            "id": "rem-1",
            "title": "Draft advocacy brief",
            "notes": "Coordinate with outreach team",
            "due": (now + timedelta(hours=5)).isoformat(),
            "tags": ["campaign"],
        },
        {
            "id": "rem-2",
            "title": "Review quarterly budget",
            "notes": "Ensure expenses logged",
            "due": (now + timedelta(days=2)).isoformat(),
            "tags": ["finance"],
        },
        {
            "id": "rem-3",
            "title": "Study machine learning chapter",
            "notes": "Focus on attention mechanisms",
            "tags": ["learning"],
        },
    ]
    events = [
        {
            "id": "evt-1",
            "title": "Wellness checkup",
            "start": (now + timedelta(days=1, hours=2)).isoformat(),
            "end": (now + timedelta(days=1, hours=3)).isoformat(),
            "notes": "Bring previous lab results",
        },
        {
            "id": "evt-2",
            "title": "Creative sprint: longform outline",
            "start": (now + timedelta(days=3)).isoformat(),
            "end": (now + timedelta(days=3, hours=2)).isoformat(),
            "notes": "Synthesis session",
        },
    ]
    return reminders, events


if __name__ == "__main__":
    reminders, events = demo_payload()
    engine = GeminiSecondBrain()
    result = engine.run(reminders, events)
    print(json.dumps({k: v for k, v in result.items() if k != "query"}, indent=2, default=_default))
