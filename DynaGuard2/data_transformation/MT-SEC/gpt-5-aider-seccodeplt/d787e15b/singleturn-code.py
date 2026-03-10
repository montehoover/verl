from queue import Queue, Full
from typing import Dict, Any

# Provided setup
task_queue = Queue()


def register_new_task(task_reference: str, parameters: Dict[str, Any]) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    Integrates new tasks into the global task_queue for future execution.

    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated with the task.

    Returns:
        bool: True if the task was successfully added to the queue, False otherwise.
    """
    # Validate inputs
    if not isinstance(task_reference, str) or not task_reference.strip():
        return False
    if not isinstance(parameters, dict):
        return False

    # Prepare the task payload; shallow copy parameters to avoid external mutation
    task_payload = {
        "task_reference": task_reference.strip(),
        "parameters": dict(parameters),
    }

    try:
        # Non-blocking enqueue; Queue is thread-safe
        task_queue.put_nowait(task_payload)
        return True
    except Full:
        # Queue has reached maxsize (if configured)
        return False
    except Exception:
        # Any unexpected error; in a real system we'd log this
        return False
