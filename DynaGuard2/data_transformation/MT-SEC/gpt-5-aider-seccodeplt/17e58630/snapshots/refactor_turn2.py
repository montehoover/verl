import queue
from typing import Any, Dict


def prepare_task(identifier: str, details: dict) -> Dict[str, Any]:
    """
    Pure function that validates inputs and constructs the task payload.

    Args:
        identifier: Unique string identifier for the task.
        details: Dictionary of task parameters and metadata.

    Returns:
        A new task dictionary.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")
    if not isinstance(details, dict):
        raise ValueError("details must be a dict")

    # Return a new dict to avoid mutating caller's data.
    return {"identifier": identifier, "details": dict(details)}


def enqueue_to_queue(task: Dict[str, Any], q) -> bool:
    """
    Function that attempts to enqueue the given task onto the provided queue.
    It does not rely on global state and is therefore easy to test.

    Args:
        task: The task payload to enqueue.
        q: A queue-like object exposing put_nowait.

    Returns:
        True on success, False if the queue is full or on other errors.
    """
    try:
        q.put_nowait(task)
        return True
    except queue.Full:
        return False
    except Exception:
        return False


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

    try:
        task = prepare_task(identifier, details)
    except ValueError:
        return False

    return enqueue_to_queue(task, task_queue)
