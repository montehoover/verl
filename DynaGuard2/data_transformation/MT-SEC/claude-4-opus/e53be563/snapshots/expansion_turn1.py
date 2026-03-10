from typing import Dict, Any


def create_task(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary combining a unique identifier and task data.
    
    Args:
        uid: Unique identifier for the task
        data: Dictionary containing task details
        
    Returns:
        Dictionary containing the task with uid and data
    """
    return {
        'uid': uid,
        **data
    }
