from typing import Dict, List
import uuid

existing_event_ids: List[str] = []
activity_log: List[Dict] = []

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

def add_event_id(user_event: Dict) -> Dict:
    """
    Assigns a unique event_id to a user event and tracks it.

    Args:
        user_event: The user event dictionary.

    Returns:
        The user event dictionary updated with an event_id.
    """
    global existing_event_ids
    # Generate a unique event ID. Using uuid for better uniqueness.
    event_id = str(uuid.uuid4())
    while event_id in existing_event_ids: # Ensure uniqueness, though highly unlikely with uuid4
        event_id = str(uuid.uuid4())
    
    user_event["event_id"] = event_id
    existing_event_ids.append(event_id)
    return user_event

def log_user_event(user_key: str, action_details: str) -> bool:
    """
    Logs a user action by creating an event, assigning an ID,
    and adding it to the activity log.

    Args:
        user_key: The identifier for the user.
        action_details: A string describing the action taken by the user.

    Returns:
        True if the event was successfully logged, False otherwise.
    """
    global activity_log
    try:
        user_event = create_user_event(user_key, action_details)
        event_with_id = add_event_id(user_event)
        activity_log.append(event_with_id)
        return True
    except Exception:
        # In a real application, more specific exception handling and logging would be appropriate.
        return False
