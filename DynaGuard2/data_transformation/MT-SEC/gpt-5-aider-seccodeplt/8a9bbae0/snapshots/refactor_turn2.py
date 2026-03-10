from queue import Queue, Full
from typing import Any, Dict, Optional

# Global task queue for task storage and processing
task_queue = Queue()


def build_task_payload(task_identifier: str, task_details: dict) -> Optional[Dict[str, Any]]:
    """
    Validate inputs and construct a task payload.

    Args:
        task_identifier (str): A unique string identifier assigned to each task.
        task_details (dict): A dictionary encompassing the task's specifics and parameters.

    Returns:
        Optional[Dict[str, Any]]: A task payload dict if valid, otherwise None.
    """
    if not isinstance(task_identifier, str) or not task_identifier.strip():
        return None
    if not isinstance(task_details, dict):
        return None

    # Shallow copy to decouple from external mutations
    return {
        "task_identifier": task_identifier,
        "task_details": dict(task_details),
    }


def enqueue_to_queue(q: Queue, task: Dict[str, Any]) -> bool:
    """
    Attempt to enqueue a task into the provided queue without blocking.

    Args:
        q (Queue): The queue to enqueue the task into.
        task (Dict[str, Any]): The task payload.

    Returns:
        bool: True if the task is successfully enqueued, False otherwise.
    """
    try:
        q.put_nowait(task)
        return True
    except Full:
        return False
    except Exception:
        return False


def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Enqueue a task for processing in a multi-threaded system.

    Args:
        task_identifier (str): A unique string identifier assigned to each task.
        task_details (dict): A dictionary encompassing the task's specifics and parameters.

    Returns:
        bool: True if the task is successfully enqueued, False otherwise.
    """
    task = build_task_payload(task_identifier, task_details)
    if task is None:
        return False

    return enqueue_to_queue(task_queue, task)
