from typing import List, Dict, Any

# Global list to store tasks as dictionaries with 'id', 'description', and 'info'
TASKS: List[Dict[str, Any]] = []

def add_task(task_id: str, task_description: str, task_info: Dict[str, Any]) -> bool:
    """
    Adds a task to the global TASKS list.
    Returns True if the task is added successfully, otherwise False.

    Parameters:
      - task_id: unique task identifier (string)
      - task_description: short description of the task (string)
      - task_info: additional details for the task (dict), e.g. {'priority': 'high', 'deadline': '2025-12-31'}

    A task is not added (returns False) if:
      - task_id or task_description are not strings
      - task_id or task_description are empty after trimming whitespace
      - task_info is not a dictionary
      - a task with the same task_id already exists
    """
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False
    if not isinstance(task_info, dict):
        return False

    tid = task_id.strip()
    desc = task_description.strip()

    if not tid or not desc:
        return False

    # Prevent duplicate task IDs
    for task in TASKS:
        if task.get("id") == tid:
            return False

    # Store the task with additional info
    TASKS.append({
        "id": tid,
        "description": desc,
        "info": dict(task_info)  # shallow copy to avoid external mutation
    })
    return True
