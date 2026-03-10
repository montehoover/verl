from typing import Dict, Any
import uuid

# Global list to track all generated task IDs
task_ids = []


def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary that encapsulates the task identifier and its content.

    Args:
        task_identifier: A unique string identifying the task.
        task_content: A dictionary containing the task's specifics.

    Returns:
        A dictionary combining the task identifier and content.
    """
    return {
        "task_identifier": task_identifier,
        "task_content": dict(task_content),  # shallow copy to avoid external mutation
    }


def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique 'task_id' to the provided task dictionary and track it globally.

    - If the task already has a 'task_id' that isn't tracked, it will be recorded.
    - If the task has no 'task_id' or it's already taken, a new unique one is generated.
    - Returns a new task dictionary (shallow copy) with 'task_id' set.

    Args:
        task: The task dictionary to update.

    Returns:
        Updated task dictionary containing a unique 'task_id'.
    """
    global task_ids
    updated_task = dict(task)  # shallow copy to avoid mutating the original input

    existing_id = updated_task.get("task_id")
    if isinstance(existing_id, str) and existing_id and existing_id not in task_ids:
        task_ids.append(existing_id)
        return updated_task

    # Generate a new UUID4-based task ID and ensure uniqueness within this runtime
    new_id = str(uuid.uuid4())
    while new_id in task_ids:
        new_id = str(uuid.uuid4())

    task_ids.append(new_id)
    updated_task["task_id"] = new_id
    return updated_task
