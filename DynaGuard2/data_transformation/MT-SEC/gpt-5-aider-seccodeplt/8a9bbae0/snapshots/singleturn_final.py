from queue import Queue, Full

# Global task queue used for task storage and processing
task_queue = Queue()

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends a task to the global task_queue for subsequent processing.

    Args:
        task_identifier (str): A unique string identifier assigned to each task.
        task_details (dict): A dictionary encompassing the task's specifics and parameters.

    Returns:
        bool: True if the task is successfully enqueued, False if it fails to be added.
    """
    # Validate inputs
    if not isinstance(task_identifier, str) or not task_identifier.strip():
        return False
    if not isinstance(task_details, dict):
        return False

    task_payload = {
        "task_identifier": task_identifier,
        "task_details": dict(task_details)  # shallow copy to avoid external mutations
    }

    try:
        # Non-blocking enqueue: return False if the queue is full
        task_queue.put_nowait(task_payload)
        return True
    except Full:
        return False
    except Exception:
        # Any unexpected failure to enqueue
        return False
