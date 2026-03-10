from queue import Queue
from typing import Dict, Any

# Global task queue used for storing and managing tasks
task_queue = Queue()

def enqueue_task(identifier: str, details: Dict[str, Any]) -> bool:
    """
    Adds a task to the global task_queue for future execution in a concurrent environment.

    Args:
        identifier (str): A unique string that serves as an identifier for each task.
        details (dict): A dictionary containing the details and parameters associated with the task.

    Returns:
        bool: True if the task was successfully added to the queue, False otherwise.
    """
    # Validate inputs
    if not isinstance(identifier, str) or not identifier.strip():
        return False
    if not isinstance(details, dict):
        return False

    # Create a task payload; shallow-copy details to decouple from external mutations
    task = {
        "identifier": identifier,
        "details": dict(details),
    }

    try:
        # Queue is thread-safe; default maxsize is infinite, so put will not block
        task_queue.put(task, block=False)
        return True
    except Exception:
        # In case of unexpected failure (e.g., queue.Full if a maxsize is later set)
        return False
