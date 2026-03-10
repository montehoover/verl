from typing import Dict
import uuid

existing_event_ids = []

def create_user_event(user_key: str, action_details: str) -> Dict[str, str]:
    return {
        "user_key": user_key,
        "action_details": action_details
    }

def add_event_id(user_event: Dict[str, str]) -> Dict[str, str]:
    event_id = str(uuid.uuid4())
    existing_event_ids.append(event_id)
    user_event['event_id'] = event_id
    return user_event
