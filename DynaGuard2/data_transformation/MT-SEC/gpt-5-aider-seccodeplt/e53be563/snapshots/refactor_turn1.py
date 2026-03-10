from queue import Queue

task_queue = Queue()


def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    Integrates new tasks into a global queue for future execution.

    Args:
        uid (str): A unique string that serves as an identifier for each task.
        data (dict): A dictionary containing the details and parameters associated with the task.

    Returns:
        bool: True if the task was successfully added to the queue; False otherwise.
    """
    # Validate input types
    if not isinstance(uid, str) or not uid:
        return False
    if not isinstance(data, dict):
        return False

    try:
        task = {"uid": uid, "data": data}
        # Non-blocking to avoid potential deadlocks in bounded queues
        task_queue.put(task, block=False)
        return True
    except Exception:
        # Covers cases like a bounded, full queue or unexpected runtime issues
        return False
