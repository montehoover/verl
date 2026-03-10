from queue import Queue, Full

# Global task queue used for storing and managing tasks
task_queue = Queue()

def push_task(uid: str, data: dict) -> bool:
    """
    Adds a task to the global task_queue for future execution.

    Parameters:
    - uid (str): A unique identifier for the task.
    - data (dict): Details and parameters of the task.

    Returns:
    - bool: True if the task was successfully queued, False otherwise.
    """
    # Basic type validation
    if not isinstance(uid, str) or uid == "":
        return False
    if not isinstance(data, dict):
        return False

    task = {"uid": uid, "data": data}

    try:
        # Non-blocking add; returns False if queue is full (for bounded queues)
        task_queue.put_nowait(task)
        return True
    except Full:
        return False
