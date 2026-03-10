import queue

def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Adds a task to the global task_queue for future execution in a thread-safe manner.

    Args:
        identifier: Unique string identifier for the task.
        details: Dictionary of task parameters and metadata.

    Returns:
        True if the task was successfully enqueued, False otherwise.
    """
    global task_queue

    if not isinstance(identifier, str) or not identifier.strip():
        return False
    if not isinstance(details, dict):
        return False

    task = {"identifier": identifier, "details": details}

    try:
        task_queue.put_nowait(task)
        return True
    except queue.Full:
        return False
    except Exception:
        return False
