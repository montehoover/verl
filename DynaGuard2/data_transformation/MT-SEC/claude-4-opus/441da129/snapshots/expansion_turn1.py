from typing import Dict

def create_user_event(user_key: str, action_details: str) -> Dict[str, str]:
    return {
        "user_key": user_key,
        "action_details": action_details
    }
