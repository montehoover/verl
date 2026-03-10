from typing import List, Dict, Any
from queue import Queue, Full

# Global list to store tasks
TASKS: List[Dict[str, Any]] = []

# Global queue to store tasks for concurrent processing
task_queue = Queue()

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

def enqueue_task(task_identifier: str, task_details: Dict[str, Any]) -> bool:
    """
    Enqueue a task for concurrent processing using a global Queue.

    Args:
        task_identifier: Unique identifier for the task (string).
        task_details: Dictionary with task specifics (e.g., priority, deadline).

    Returns:
        True if the task was enqueued successfully, otherwise False.
    """
    # Validate inputs
    if not isinstance(task_identifier, str) or not isinstance(task_details, dict):
        return False

    task_identifier = task_identifier.strip()
    if not task_identifier:
        return False

    # Prepare payload; copy details to avoid external mutation issues
    try:
        payload: Dict[str, Any] = {
            "id": task_identifier,
            "details": dict(task_details),
        }
    except Exception:
        return False

    # Enqueue without blocking
    try:
        task_queue.put_nowait(payload)
        return True
    except Full:
        return False
    except Exception:
        return False
