from typing import Dict
from datetime import datetime, timezone


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
