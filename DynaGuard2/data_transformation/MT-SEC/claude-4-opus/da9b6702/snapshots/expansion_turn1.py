from typing import Dict

def create_user_event(user_identifier: str, task: str) -> Dict[str, str]:
    return {
        "user_identifier": user_identifier,
        "task": task
    }
