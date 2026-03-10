from typing import Dict, List
import uuid

existing_event_ids: List[str] = []

def create_user_event(user_identifier: str, task: str) -> Dict[str, str]:
    """
    Captures a user action and returns it as a dictionary.

    Args:
        user_identifier: The identifier for the user.
        task: A description of the user's action.

    Returns:
        A dictionary containing the user_identifier and task.
    """
    return {
        "user_identifier": user_identifier,
        "task": task,
    }

def assign_event_id(event: Dict[str, str]) -> Dict[str, str]:
    """
    Assigns a unique event_id to an event and tracks it.

    Args:
        event: The event dictionary.

    Returns:
        The event dictionary updated with an event_id.
    """
    global existing_event_ids
    # Generate a unique event ID. Using uuid for better uniqueness.
    event_id = str(uuid.uuid4())
    event["event_id"] = event_id
    existing_event_ids.append(event_id)
    return event
