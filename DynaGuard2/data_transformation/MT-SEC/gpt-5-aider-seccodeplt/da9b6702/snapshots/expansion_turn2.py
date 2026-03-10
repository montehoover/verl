from typing import Dict
import uuid

existing_event_ids: list[str] = []

def create_user_event(user_identifier: str, task: str) -> Dict[str, str]:
    return {"user_identifier": user_identifier, "task": task}

def assign_event_id(event: Dict[str, str]) -> Dict[str, str]:
    event_with_id = dict(event)
    new_id = uuid.uuid4().hex
    while new_id in existing_event_ids:
        new_id = uuid.uuid4().hex
    event_with_id["event_id"] = new_id
    existing_event_ids.append(new_id)
    return event_with_id
