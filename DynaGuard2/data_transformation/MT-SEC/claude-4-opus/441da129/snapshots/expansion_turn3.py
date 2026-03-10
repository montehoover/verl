from typing import Dict
import uuid

existing_event_ids = []
activity_log = []

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

def log_user_event(user_key: str, action_details: str) -> bool:
    try:
        user_event = create_user_event(user_key, action_details)
        user_event_with_id = add_event_id(user_event)
        activity_log.append(user_event_with_id)
        return True
    except:
        return False
