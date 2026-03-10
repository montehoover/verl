from typing import List, Dict

# Global in-memory task storage (simple list as requested)
TASKS: List[Dict[str, str]] = []

def add_task(task_id: str, task_description: str) -> bool:
    """
    Add a task to the global TASKS list.
    Returns True if added successfully, otherwise False.

    Rules:
    - Both task_id and task_description must be strings.
    - task_id and task_description must be non-empty after trimming.
    - task_id must be unique (no duplicate IDs).
    """
    global TASKS

    # Validate types
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False

    task_id = task_id.strip()
    task_description = task_description.strip()

    # Validate content
    if not task_id or not task_description:
        return False

    # Ensure unique task_id
    if any(task.get("id") == task_id for task in TASKS):
        return False

    TASKS.append({"id": task_id, "description": task_description})
    return True
