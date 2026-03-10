"""
Task queue utilities for a parallel processing system.

This module exposes a high-level insert_task_to_queue function that validates
incoming job data and enqueues a prepared task into a global, shared queue.
Helper functions are provided to keep responsibilities separated and the code
maintainable.
"""

from queue import Full, Queue
from typing import Any


# Globally accessible queue for task management and storage.
task_queue: Queue = Queue()


def prepare_task(job_id: str, job_data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Validate inputs and construct a normalized task payload.

    This is a pure function with no side effects. It ensures that the provided
    job identifier and associated data meet basic validity requirements before
    creating a task dictionary suitable for enqueuing.

    Args:
        job_id (str): Unique identifier for the task. Must be a non-empty string.
        job_data (dict[str, Any]): Details and parameters for the task.

    Returns:
        dict[str, Any] | None: The prepared task dictionary if inputs are valid;
        otherwise, None.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        return None

    if not isinstance(job_data, dict):
        return None

    return {
        "job_id": job_id,
        "job_data": job_data,
    }


def enqueue_task(queue: Queue, task: dict[str, Any]) -> bool:
    """
    Insert a prepared task into the provided queue in a non-blocking manner.

    This function attempts to place the task into the target queue immediately.
    If the queue is bounded and currently full, the operation fails gracefully.

    Args:
        queue (Queue): The target queue used for task storage.
        task (dict[str, Any]): The already validated/prepared task payload.

    Returns:
        bool: True if insertion succeeded; False if the queue is full or any
        unexpected error occurs.
    """
    try:
        queue.put_nowait(task)
        return True
    except Full:
        return False
    except Exception:
        return False


def insert_task_to_queue(job_id: str, job_data: dict[str, Any]) -> bool:
    """
    Facilitate task insertion into the shared, global queue.

    This function is the primary entry point: it validates the inputs, prepares
    a task payload, and enqueues it into the globally accessible task_queue.

    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict[str, Any]): A dictionary encompassing the task's details,
            parameters, and other relevant information.

    Returns:
        bool: True if the task was successfully added to the queue; False otherwise.
    """
    task = prepare_task(job_id, job_data)
    if task is None:
        return False

    return enqueue_task(task_queue, task)
