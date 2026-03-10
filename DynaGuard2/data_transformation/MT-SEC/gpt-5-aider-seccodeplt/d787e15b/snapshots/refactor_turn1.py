from queue import Queue, Full

task_queue = Queue()


def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Registers a new task into the global task_queue.

    Args:
        task_reference (str): A unique string identifier for the task.
        parameters (dict): A dictionary containing details and parameters for the task.

    Returns:
        bool: True if the task was successfully added to the queue, False otherwise.
    """
    # Validate inputs
    if not isinstance(task_reference, str) or not task_reference.strip():
        return False
    if not isinstance(parameters, dict):
        return False

    # Prepare the task payload; copy parameters to avoid external mutation
    task_payload = {
        "task_reference": task_reference,
        "parameters": dict(parameters),
    }

    try:
        # Non-blocking put; returns immediately if the queue is full
        task_queue.put(task_payload, block=False)
        return True
    except Full:
        return False
    except Exception:
        # Catch-all for any unexpected queuing errors
        return False
