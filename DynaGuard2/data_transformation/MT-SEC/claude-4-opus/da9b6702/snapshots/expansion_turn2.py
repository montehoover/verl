from typing import Dict
import uuid

existing_event_ids = []

def create_user_event(user_identifier: str, task: str) -> Dict[str, str]:
    return {
        "user_identifier": user_identifier,
        "task": task
    }

def assign_event_id(event: Dict[str, str]) -> Dict[str, str]:
    event_id = str(uuid.uuid4())
    existing_event_ids.append(event_id)
    event['event_id'] = event_id
    return event
