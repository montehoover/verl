"""
Task queue insertion utility.

Exposes a thread-safe function, insert_task_to_queue, that enqueues tasks
into the global task_queue for later processing.
"""

from queue import Queue, Full

# Globally accessible queue utilized for task storage and processing
task_queue = Queue()


def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Append a new task to the global task queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    # Basic validation of inputs
    if not isinstance(task_identifier, str) or not task_identifier:
        return False
    if not isinstance(task_content, dict):
        return False

    task = {
        "task_identifier": task_identifier,
        "task_content": task_content,
    }

    try:
        # Queue is thread-safe; put_nowait lets us detect a bounded queue's full state.
        task_queue.put_nowait(task)
        return True
    except Full:
        return False
    except Exception:
        # Any unexpected error results in failure signal
        return False
