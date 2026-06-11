"""Shared utilities for building and evaluating JUSThink HumanLM data."""

from __future__ import annotations

import copy
import re
from collections import Counter


HUMAN_PARTICIPANTS = {"A", "B"}
ACTION_VERBS = {"adds", "removes", "loads", "presses", "submits"}
EDIT_VERBS = {"adds", "removes", "loads"}

CPS_STATE_DIMENSIONS = {
    "belief": {
        "cps_name": "task_understanding",
        "description": "What the student currently believes about the graph, costs, constraints, teammate knowledge, and submission feedback.",
    },
    "goal": {
        "cps_name": "strategy_goal",
        "description": "What the student is trying to achieve next, including route construction, information gathering, checking, repair, or submission.",
    },
    "value": {
        "cps_name": "collaboration_value",
        "description": "What the student prioritizes in collaboration, such as speed, certainty, low cost, shared understanding, or following the teammate.",
    },
    "stance": {
        "cps_name": "interaction_stance",
        "description": "The student's position toward the teammate's proposal or current plan, such as agreement, disagreement, uncertainty, or challenge.",
    },
    "emotion": {
        "cps_name": "error_repair_state",
        "description": "The student's affective and repair state, such as confidence, confusion, frustration, urgency, or readiness to correct an error.",
    },
    "communication": {
        "cps_name": "communication_style",
        "description": "How the student communicates, such as instruction, query, confirmation, explanation, repair talk, or concise acknowledgment.",
    },
}


def normalize_edge(value: object) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return sorted([int(value[0]), int(value[1])])
    match = re.search(r"\((\d+)-(\d+)\)", str(value))
    if match:
        return sorted([int(match.group(1)), int(match.group(2))])
    return None


def matching_label(event: dict) -> str | None:
    text = str(event.get("matching", ""))
    if "MISMATCH_" in text:
        return "mismatch"
    if "NONMATCH_" in text:
        return "nonmatch"
    if "MATCH_" in text:
        return "match"
    return None


def action_type(event: dict) -> str | None:
    verb = event.get("verb")
    obj = str(event.get("object", "")).lower()
    if verb in EDIT_VERBS:
        return {"adds": "edit_add", "removes": "edit_remove", "loads": "edit_load"}[verb]
    if verb == "submits" or (verb == "presses" and "submit" in obj):
        return "submit"
    if verb == "presses" and any(token in obj for token in ("compare", "check", "previous", "next")):
        return "check"
    if verb == "presses":
        return "press"
    return None


def discourse_type(event: dict) -> str | None:
    if event.get("verb") != "says":
        return None
    text = str(event.get("object", "")).strip().lower()
    if not text:
        return None
    if any(token in text for token in ("submit", "send it", "try it")):
        return "submit_proposal"
    if any(token in text for token in ("cost", "price", "cheap", "expensive")):
        return "cost_query" if "?" in text or text.startswith(("what", "how")) else "cost_report"
    if any(token in text for token in ("wrong", "mistake", "sorry", "remove", "delete", "fix")):
        return "repair_talk"
    if any(token in text for token in ("no ", "not ", "don't", "doesn't", "but ")):
        return "disagreement"
    if text.startswith(("add ", "go ", "connect ", "put ", "take ", "use ")):
        return "route_instruction"
    if any(token in text for token in ("yes", "yeah", "okay", "ok", "right", "sure")):
        return "confirmation"
    if any(token in text for token in ("failed", "error", "feedback", "optimal")):
        return "feedback_interpretation"
    if len(text.split()) <= 3:
        return "filler_or_hesitation"
    return "other_utterance"


def serialize_event(event: dict) -> dict:
    result = {
        "attempt_no": event.get("attempt_no"),
        "turn_no": event.get("turn_no"),
        "subject": event.get("subject"),
        "verb": event.get("verb"),
        "object": event.get("object"),
    }
    for key in ("instructions", "pending_instructions", "matching", "submit_result"):
        value = event.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    label = matching_label(event)
    if label:
        result["matching_label"] = label
    discourse = discourse_type(event)
    if discourse:
        result["discourse_type"] = discourse
    return result


def prefix_environment_state(events: list[dict]) -> dict:
    edges: set[tuple[int, int]] = set()
    latest_submit_feedback = None
    submission_results: list[dict] = []
    pending_instructions = None

    for event in events:
        verb = event.get("verb")
        if verb == "adds":
            edge = normalize_edge(event.get("object"))
            if edge:
                edges.add(tuple(edge))
        elif verb == "removes":
            edge = normalize_edge(event.get("object"))
            if edge:
                edges.discard(tuple(edge))
        elif verb == "loads" and isinstance(event.get("object"), list):
            loaded = [normalize_edge(edge) for edge in event["object"]]
            edges = {tuple(edge) for edge in loaded if edge}

        if event.get("pending_instructions") not in (None, "", []):
            pending_instructions = event.get("pending_instructions")

        submit = event.get("submit_result")
        if submit and event.get("subject") == "T":
            latest_submit_feedback = copy.deepcopy(submit)
            submission_results.append(copy.deepcopy(submit))

    best_gap = min(
        (float(result["abs_error"]) for result in submission_results if result.get("abs_error") is not None),
        default=None,
    )
    return {
        "current_edges": [list(edge) for edge in sorted(edges)],
        "pending_instructions": pending_instructions or [],
        "latest_submit_feedback": latest_submit_feedback,
        "performance_so_far": {
            "submission_count": len(submission_results),
            "best_cost_gap": best_gap,
            "found_optimal_so_far": any(
                float(result.get("abs_error", 1)) == 0 for result in submission_results
            ),
            "cost_gap_trajectory": [
                result.get("abs_error")
                for result in submission_results
                if result.get("abs_error") is not None
            ],
        },
    }


def prefix_role_state(events: list[dict], participant: str) -> dict:
    participant_events = [event for event in events if event.get("subject") == participant]
    return {
        "declared_role": "not_available_in_bundle",
        "observed_capabilities_so_far": {
            "has_edited": any(event.get("verb") in EDIT_VERBS for event in participant_events),
            "has_checked_or_compared": any(
                action_type(event) == "check" for event in participant_events
            ),
            "has_submitted": any(action_type(event) == "submit" for event in participant_events),
        },
        "note": "The bundle does not reliably encode current cost-view/edit-view assignment.",
    }


def prefix_student_profile(events: list[dict], participant: str) -> dict:
    participant_events = [event for event in events if event.get("subject") == participant]
    utterances = [
        str(event.get("object", "")).strip()
        for event in participant_events
        if event.get("verb") == "says" and str(event.get("object", "")).strip()
    ]
    actions = [action_type(event) for event in participant_events]
    matches = Counter(
        label for event in participant_events if (label := matching_label(event))
    )
    word_counts = [len(text.split()) for text in utterances]
    return {
        "source": "prefix_behavior_only",
        "participant": participant,
        "utterance_count_so_far": len(utterances),
        "average_utterance_words_so_far": (
            round(sum(word_counts) / len(word_counts), 3) if word_counts else None
        ),
        "action_counts_so_far": dict(Counter(action for action in actions if action)),
        "instruction_alignment_counts_so_far": dict(matches),
    }


def target_actions_after_utterance(events: list[dict], idx: int, participant: str) -> list[dict]:
    actions = []
    for event in events[idx + 1 :]:
        if event.get("verb") == "says" and event.get("subject") in HUMAN_PARTICIPANTS:
            break
        if event.get("subject") == participant and action_type(event):
            actions.append(serialize_event(event))
    return actions


def build_proxy_labels(actions: list[dict]) -> dict:
    labels = Counter()
    for event in actions:
        kind = action_type(event)
        if kind:
            labels[kind] += 1
        match = event.get("matching_label") or matching_label(event)
        if match:
            labels[match] += 1
    return dict(labels)
