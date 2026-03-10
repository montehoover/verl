from typing import Dict

def create_user_event(user_key: str, action_details: str) -> Dict:
    """
    Encapsulates user actions into a structured format.

    Args:
        user_key: The identifier for the user.
        action_details: A string describing the action taken by the user.

    Returns:
        A dictionary representing the user event.
    """
    return {
        "user_key": user_key,
        "action_details": action_details,
    }
