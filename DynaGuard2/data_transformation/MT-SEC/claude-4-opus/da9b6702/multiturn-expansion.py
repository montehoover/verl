from typing import Dict
import uuid

existing_event_ids = []
activity_log = []

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

def save_user_event(user_identifier: str, task: str) -> bool:
    try:
        event = create_user_event(user_identifier, task)
        event_with_id = assign_event_id(event)
        activity_log.append(event_with_id)
        return True
    except:
        return False
