from queue import Queue, Full

# Global task queue for task storage and processing
task_queue = Queue()


def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Enqueue a task for processing in a multi-threaded system.

    Args:
        task_identifier (str): A unique string identifier assigned to each task.
        task_details (dict): A dictionary encompassing the task's specifics and parameters.

    Returns:
        bool: True if the task is successfully enqueued, False otherwise.
    """
    # Validate inputs
    if not isinstance(task_identifier, str) or not task_identifier.strip():
        return False
    if not isinstance(task_details, dict):
        return False

    # Compose the task payload
    task = {
        "task_identifier": task_identifier,
        "task_details": task_details,
    }

    # Attempt to enqueue without blocking
    try:
        task_queue.put_nowait(task)
        return True
    except Full:
        return False
    except Exception:
        # In case of any unexpected error, signal failure
        return False
