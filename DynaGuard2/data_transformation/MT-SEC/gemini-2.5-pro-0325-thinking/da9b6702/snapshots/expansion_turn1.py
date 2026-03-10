from typing import Dict

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
