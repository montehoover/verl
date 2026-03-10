"""
Utilities for registering tasks into a shared, thread-safe queue.

This module provides a small API centered around `register_new_task`, which
validates input, prepares a task payload, and enqueues it for later processing.
"""

from queue import Full, Queue
from typing import Any, Dict


# Type aliases for clarity.
TaskPayload = Dict[str, Any]


# Global task queue used across the application.
task_queue = Queue()


def validate_inputs(task_reference: str, parameters: dict) -> bool:
    """
    Validate input arguments for registering a new task.

    The function ensures that:
    - task_reference is a non-empty string (ignoring surrounding whitespace).
    - parameters is a dictionary.

    Args:
        task_reference (str): Unique identifier for the task. Must be non-empty
            after trimming whitespace.
        parameters (dict): Dictionary containing task parameters and metadata.

    Returns:
        bool: True if inputs are valid; False otherwise.

    Examples:
        >>> validate_inputs("task-123", {"a": 1})
        True
        >>> validate_inputs("", {"a": 1})
        False
        >>> validate_inputs("task-123", ["not-a-dict"])
        False
    """
    if not isinstance(task_reference, str) or not task_reference.strip():
        return False

    if not isinstance(parameters, dict):
        return False

    return True


def prepare_task_payload(task_reference: str, parameters: dict) -> TaskPayload:
    """
    Prepare the task payload for queueing.

    Creates a shallow copy of the provided parameters to avoid external
    mutation affecting queued tasks.

    Args:
        task_reference (str): Unique identifier for the task.
        parameters (dict): Dictionary containing task parameters and metadata.

    Returns:
        TaskPayload: A dictionary representing the task payload, containing:
            - "task_reference": str
            - "parameters": dict

    Examples:
        >>> prepare_task_payload("task-123", {"retries": 3})
        {'task_reference': 'task-123', 'parameters': {'retries': 3}}
    """
    payload: TaskPayload = {
        "task_reference": task_reference,
        "parameters": dict(parameters),
    }

    return payload


def enqueue_task(
    target_queue: Queue,
    payload: TaskPayload,
    block: bool = False,
) -> bool:
    """
    Attempt to enqueue the provided payload into the given queue.

    The underlying Queue is thread-safe. By default, this function uses a
    non-blocking put to avoid stalling producers.

    Args:
        target_queue (Queue): The queue to insert the payload into.
        payload (TaskPayload): The task payload to enqueue.
        block (bool): Whether to block if the queue is full. Defaults to False.

    Returns:
        bool: True if enqueued successfully; False otherwise (e.g., if the
        queue is full in non-blocking mode, or an unexpected error occurs).

    Examples:
        >>> q = Queue(maxsize=1)
        >>> enqueue_task(q, {"task_reference": "t", "parameters": {}}, block=False)
        True
    """
    try:
        target_queue.put(payload, block=block)
        return True
    except Full:
        return False
    except Exception:
        # Catch-all for any unexpected queuing errors
        return False


def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Register a new task into the global task_queue.

    This is the primary entry point for producers. It validates the inputs,
    prepares a normalized task payload, and enqueues it into the shared queue.

    Concurrency:
        The global `task_queue` is an instance of `queue.Queue`, which is
        thread-safe and suitable for multi-producer, multi-consumer patterns.

    Args:
        task_reference (str): A unique string identifier for the task.
        parameters (dict): A dictionary containing details and parameters for
            the task.

    Returns:
        bool: True if the task was successfully added to the queue; False
        otherwise.

    Examples:
        >>> register_new_task("task-001", {"priority": "high"})
        True
    """
    if not validate_inputs(task_reference, parameters):
        return False

    task_payload = prepare_task_payload(task_reference, parameters)

    return enqueue_task(task_queue, task_payload, block=False)
