from typing import List, Dict

# Global list to store tasks as dictionaries with 'id' and 'description'
TASKS: List[Dict[str, str]] = []

def add_task(task_id: str, task_description: str) -> bool:
    """
    Adds a task to the global TASKS list.
    Returns True if the task is added successfully, otherwise False.

    A task is not added (returns False) if:
      - task_id or task_description are not strings
      - task_id or task_description are empty after trimming whitespace
      - a task with the same task_id already exists
    """
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False

    tid = task_id.strip()
    desc = task_description.strip()

    if not tid or not desc:
        return False

    # Prevent duplicate task IDs
    for task in TASKS:
        if task.get("id") == tid:
            return False

    TASKS.append({"id": tid, "description": desc})
    return True
