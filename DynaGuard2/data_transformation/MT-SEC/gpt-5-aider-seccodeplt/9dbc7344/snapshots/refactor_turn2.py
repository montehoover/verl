from queue import Queue, Full
from typing import Any, Dict
from copy import deepcopy

# Globally accessible queue utilized for task storage and processing
task_queue: Queue = Queue()


def _validate_task_identifier(task_identifier: str) -> bool:
    """Pure validation for task identifier."""
    return isinstance(task_identifier, str) and bool(task_identifier)


def _validate_task_content(task_content: Dict[str, Any]) -> bool:
    """Pure validation for task content."""
    return isinstance(task_content, dict)


def create_task_payload(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function to create a normalized task payload.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        A dictionary representing the task payload.
    """
    return {
        "id": task_identifier,
        "content": deepcopy(task_content),
    }


def enqueue_task_payload(queue: Queue, task: Dict[str, Any]) -> bool:
    """
    Attempts to enqueue the task payload into the provided queue without blocking.

    Args:
        queue: The queue to which the task will be added.
        task: The task payload.

    Returns:
        True if enqueued successfully, False otherwise.
    """
    try:
        queue.put(task, block=False)
        return True
    except Full:
        return False
    except Exception:
        return False


def insert_task_to_queue(task_identifier: str, task_content: Dict[str, Any]) -> bool:
    """
    Appends a new task to the global queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    # Validation stage
    if not _validate_task_identifier(task_identifier):
        return False
    if not _validate_task_content(task_content):
        return False

    # Construction stage
    task = create_task_payload(task_identifier, task_content)

    # Enqueue stage
    return enqueue_task_payload(task_queue, task)
