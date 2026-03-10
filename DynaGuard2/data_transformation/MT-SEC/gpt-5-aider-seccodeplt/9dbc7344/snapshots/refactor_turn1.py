from queue import Queue, Full
from typing import Any, Dict

# Globally accessible queue utilized for task storage and processing
task_queue: Queue = Queue()


def insert_task_to_queue(task_identifier: str, task_content: Dict[str, Any]) -> bool:
    """
    Appends a new task to the global queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    # Validate inputs
    if not isinstance(task_identifier, str) or not task_identifier:
        return False
    if not isinstance(task_content, dict):
        return False

    # Normalize task payload
    task = {
        "id": task_identifier,
        "content": task_content,
    }

    # Enqueue without blocking
    try:
        task_queue.put(task, block=False)
        return True
    except Full:
        # Queue has reached its maximum size (if configured)
        return False
    except Exception:
        return False
