from typing import List, Dict

# Global list to store tasks
TASKS: List[Dict[str, str]] = []

def add_task(task_id: str, task_description: str) -> bool:
    """
    Add a task to the global TASKS list.

    Args:
        task_id: Unique identifier for the task (string).
        task_description: Description of the task (string).

    Returns:
        True if the task was added successfully, otherwise False.
    """
    # Validate types
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False

    # Normalize and validate content
    task_id = task_id.strip()
    task_description = task_description.strip()
    if not task_id or not task_description:
        return False

    # Ensure task_id is unique
    if any(t["id"] == task_id for t in TASKS):
        return False

    # Add the task
    TASKS.append({"id": task_id, "description": task_description})
    return True
