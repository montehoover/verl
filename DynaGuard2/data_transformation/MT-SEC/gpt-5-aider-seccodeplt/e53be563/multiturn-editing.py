from typing import List, Dict, Any
from queue import Queue

# Global in-memory task storage (simple list as requested)
TASKS: List[Dict[str, Any]] = []

# Global queue for concurrent task handling
task_queue = Queue()

def add_task(task_id: str, task_description: str, task_info: Dict[str, Any]) -> bool:
    """
    Add a task to the global TASKS list with additional info.
    Returns True if added successfully, otherwise False.

    Rules:
    - task_id and task_description must be strings and non-empty after trimming.
    - task_info must be a dictionary (e.g., keys like 'priority', 'deadline').
    - task_id must be unique (no duplicate IDs).
    """
    global TASKS

    # Validate types
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False
    if not isinstance(task_info, dict):
        return False

    task_id = task_id.strip()
    task_description = task_description.strip()

    # Validate content
    if not task_id or not task_description:
        return False

    # Ensure unique task_id
    if any(task.get("id") == task_id for task in TASKS):
        return False

    # Shallow copy task_info to prevent external mutation side-effects
    try:
        info_copy = dict(task_info)
    except Exception:
        return False

    TASKS.append({"id": task_id, "description": task_description, "info": info_copy})
    return True

def push_task(uid: str, data: Dict[str, Any]) -> bool:
    """
    Push a task into the global concurrent queue.

    Parameters:
    - uid: unique task identifier (string, non-empty after trimming)
    - data: dictionary with task parameters

    Returns:
    - True if the task is successfully enqueued, False otherwise
    """
    # Validate input types
    if not isinstance(uid, str) or not isinstance(data, dict):
        return False

    uid = uid.strip()
    if not uid:
        return False

    # Make a shallow copy to avoid external mutation
    try:
        payload = {"id": uid, "data": dict(data)}
    except Exception:
        return False

    try:
        task_queue.put(payload, block=False)
        return True
    except Exception:
        # Covers cases like queue being full if bounded, or unexpected errors
        return False
