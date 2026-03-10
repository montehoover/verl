from typing import List, Dict, Any

# Global list to store tasks
TASKS: List[Dict[str, Any]] = []

def add_task(task_id: str, task_description: str, task_info: Dict[str, str]) -> bool:
    """
    Add a task to the global TASKS list.

    Args:
        task_id: Unique identifier for the task (string).
        task_description: Description of the task (string).
        task_info: A dictionary containing additional task details (e.g., 'priority', 'deadline').

    Returns:
        True if the task was added successfully, otherwise False.
    """
    # Validate basic types
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False
    if not isinstance(task_info, dict):
        return False

    # Normalize and validate content
    task_id = task_id.strip()
    task_description = task_description.strip()
    if not task_id or not task_description:
        return False

    # Validate and normalize task_info (keys and values must be strings)
    normalized_info: Dict[str, str] = {}
    for k, v in task_info.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return False
        k_norm = k.strip()
        v_norm = v.strip()
        if not k_norm:
            return False
        normalized_info[k_norm] = v_norm

    # Ensure task_id is unique
    if any(t["id"] == task_id for t in TASKS):
        return False

    # Add the task with additional info
    TASKS.append({
        "id": task_id,
        "description": task_description,
        "info": normalized_info
    })
    return True
