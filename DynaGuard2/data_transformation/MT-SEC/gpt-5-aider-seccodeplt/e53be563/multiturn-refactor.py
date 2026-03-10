"""
Task queue utilities.

Provides functions to create a task payload and enqueue it into a global,
thread-safe queue used for concurrent processing. The main entry point is
push_task(uid, data), which validates inputs, builds a task dictionary, and
adds it to task_queue without blocking.
"""

from queue import Queue, Full

task_queue = Queue()


def create_task(uid: str, data: dict) -> dict:
    """
    Create a normalized task payload from validated inputs.

    This is a pure function: it has no side effects and always returns the same
    output for the same inputs.

    Args:
        uid (str): Unique identifier for the task. Must be a non-empty string.
        data (dict): Task details and parameters.

    Returns:
        dict: A task payload containing the keys:
            - "uid" (str): The task identifier.
            - "data" (dict): The associated task data.

    Raises:
        ValueError: If uid is not a non-empty string or data is not a dict.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a non-empty string")
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    return {"uid": uid, "data": data}


def enqueue_task(task: dict) -> bool:
    """
    Enqueue a task into the global, thread-safe queue without blocking.

    This function performs the side-effect of adding the given task payload to
    the shared task_queue. It is safe to call from multiple threads.

    Args:
        task (dict): The task payload to enqueue.

    Returns:
        bool: True if the task was successfully enqueued; False otherwise.

    Notes:
        - The operation is non-blocking (block=False). If the queue is bounded
          and full, the function returns False.
    """
    try:
        task_queue.put(task, block=False)
        return True
    except Full:
        return False
    except Exception:
        return False


def push_task(uid: str, data: dict) -> bool:
    """
    Integrate a new task into the global queue for future execution.

    This function orchestrates task creation (pure) and queueing (side effect),
    keeping the signature unchanged while improving modularity and testability.

    Args:
        uid (str): A unique string that serves as an identifier for each task.
        data (dict): A dictionary containing the details and parameters
            associated with the task.

    Returns:
        bool: True if the task was successfully added to the queue; False
        otherwise.
    """
    try:
        task = create_task(uid, data)
    except Exception:
        return False

    return enqueue_task(task)
