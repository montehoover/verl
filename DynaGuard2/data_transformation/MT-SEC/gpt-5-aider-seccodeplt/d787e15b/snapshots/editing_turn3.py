from typing import List, Dict, Any, Set
from threading import Lock
from queue import Queue

# Global list to store tasks as dictionaries with 'id', 'description', and 'info'
TASKS: List[Dict[str, Any]] = []

# Global queue to handle tasks in a concurrent environment
task_queue = Queue()

# Internal registry to prevent duplicate task IDs in the queue
_task_registry_lock = Lock()
_registered_task_ids: Set[str] = set()

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


def register_new_task(task_reference: str, parameters: Dict[str, Any]) -> bool:
    """
    Registers a new task into the global task_queue for concurrent processing.
    Ensures each task_reference is unique (not already enqueued).
    Returns True on success, False on validation failure or if already registered.

    Parameters:
      - task_reference: unique task identifier (string)
      - parameters: additional task details (dict), e.g. {'priority': 'high', 'deadline': '2025-12-31'}
    """
    if not isinstance(task_reference, str) or not isinstance(parameters, dict):
        return False

    tref = task_reference.strip()
    if not tref:
        return False

    # Ensure uniqueness and enqueue atomically with respect to registration
    with _task_registry_lock:
        if tref in _registered_task_ids:
            return False
        # Reserve the ID
        _registered_task_ids.add(tref)

    try:
        # Enqueue the task payload
        task_queue.put_nowait({
            "id": tref,
            "parameters": dict(parameters)  # shallow copy to avoid external mutation
        })
        return True
    except Exception:
        # Rollback reservation on failure to enqueue
        with _task_registry_lock:
            _registered_task_ids.discard(tref)
        return False
