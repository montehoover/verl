"""
Utilities for preparing and enqueuing tasks using a simple pipeline pattern.

This module exposes three functions:
- prepare_task: Pure function that validates inputs and constructs a task
  payload.
- enqueue_to_queue: Side-effect function that enqueues a prepared task to a
  provided queue-like object.
- enqueue_task: Public API that composes the two prior steps using the
  globally available 'task_queue'.

The design separates pure transformation from side-effectful operations,
improving readability and testability.
"""

import queue
from typing import Any, Dict


def prepare_task(identifier: str, details: dict) -> Dict[str, Any]:
    """
    Validate inputs and construct a task payload.

    This is a pure function: it performs validation and returns a new task
    dictionary without mutating inputs or relying on global state.

    Args:
        identifier:
            Unique, non-empty string identifier for the task.
        details:
            Dictionary of task parameters and metadata. Must be a plain dict.

    Returns:
        Dict[str, Any]: A new task dictionary with keys:
            - "identifier": str
            - "details": dict (shallow-copied)

    Raises:
        ValueError: If 'identifier' is not a non-empty string or if 'details'
            is not a dict.

    Examples:
        >>> prepare_task("task-1", {"priority": "high"})
        {'identifier': 'task-1', 'details': {'priority': 'high'}}
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")
    if not isinstance(details, dict):
        raise ValueError("details must be a dict")

    # Return a new dict to avoid mutating caller's data.
    return {"identifier": identifier, "details": dict(details)}


def enqueue_to_queue(task: Dict[str, Any], task_queue_obj: queue.Queue) -> bool:
    """
    Enqueue the given task onto the provided queue in a non-blocking manner.

    This function performs side effects but does not rely on global state,
    which makes it straightforward to test by passing in a queue-like object
    that exposes 'put_nowait'.

    Args:
        task:
            The task payload to enqueue. Typically the output of
            'prepare_task'.
        task_queue_obj:
            A queue-like object exposing 'put_nowait'. Usually a
            'queue.Queue' instance.

    Returns:
        bool: True on success; False if the queue is full or on other errors.

    Examples:
        >>> from queue import Queue
        >>> q = Queue()
        >>> enqueue_to_queue({'identifier': 'x', 'details': {}}, q)
        True
    """
    try:
        task_queue_obj.put_nowait(task)
        return True
    except queue.Full:
        return False
    except Exception:
        return False


def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Add a task to the global 'task_queue' for future execution.

    This function composes the pipeline:
      1) Validate and prepare the task payload (pure step).
      2) Enqueue the prepared payload (side-effect step).

    It expects a global variable named 'task_queue' to exist in the runtime
    environment (e.g., defined as 'task_queue = queue.Queue()').

    Args:
        identifier:
            Unique, non-empty string identifier for the task.
        details:
            Dictionary of task parameters and metadata.

    Returns:
        bool: True if the task was successfully enqueued; otherwise False.

    Examples:
        >>> from queue import Queue
        >>> global task_queue
        >>> task_queue = Queue()
        >>> enqueue_task("task-42", {"foo": "bar"})
        True
    """
    global task_queue

    try:
        task = prepare_task(identifier, details)
    except ValueError:
        return False

    return enqueue_to_queue(task, task_queue)
