from typing import Dict
from datetime import datetime, timezone
from uuid import uuid4

existing_event_ids = []
activity_log = []


def create_user_event(user_key: str, action_details: str) -> Dict[str, str]:
    if not isinstance(user_key, str) or not user_key:
        raise ValueError("user_key must be a non-empty string")
    if not isinstance(action_details, str) or not action_details:
        raise ValueError("action_details must be a non-empty string")

    event: Dict[str, str] = {
        "event_type": "user_action",
        "user_key": user_key,
        "action_details": action_details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return event


def add_event_id(event: Dict[str, str]) -> Dict[str, str]:
    if not isinstance(event, dict):
        raise ValueError("event must be a dictionary")

    existing = event.get("event_id")
    if isinstance(existing, str) and existing:
        if existing not in existing_event_ids:
            existing_event_ids.append(existing)
        return event

    # Generate a unique event_id
    event_id = uuid4().hex
    while event_id in existing_event_ids:
        event_id = uuid4().hex

    event["event_id"] = event_id
    existing_event_ids.append(event_id)
    return event


def log_user_event(user_key: str, action_details: str) -> bool:
    try:
        event = create_user_event(user_key, action_details)
        event = add_event_id(event)
        activity_log.append(event)
        return True
    except Exception:
        return False
