"""
Task queuing utilities for a multi-threaded system.

This module exposes a primary API function, insert_task_to_queue, which
validates inputs, constructs a normalized task payload, and enqueues it
to a globally accessible queue for later processing.

The implementation follows a simple pipeline pattern:
    validate -> construct payload -> enqueue
"""

from copy import deepcopy
import logging
from queue import Full, Queue
from typing import Any, Dict

# Configure basic logging only if the application hasn't configured it yet.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)

# Globally accessible queue utilized for task storage and processing.
task_queue: Queue = Queue()


def _validate_task_identifier(task_identifier: str) -> bool:
    """
    Pure validation for task identifier.

    Args:
        task_identifier: The proposed unique task identifier.

    Returns:
        True when the identifier is a non-empty string, False otherwise.
    """
    return isinstance(task_identifier, str) and bool(task_identifier)


def _validate_task_content(task_content: Dict[str, Any]) -> bool:
    """
    Pure validation for task content.

    Args:
        task_content: The proposed task content/payload.

    Returns:
        True when the content is a dict, False otherwise.
    """
    return isinstance(task_content, dict)


def create_task_payload(
    task_identifier: str, task_content: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Pure function to create a normalized task payload.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and
            parameters.

    Returns:
        A dictionary representing the task payload.
    """
    # Use deepcopy to avoid accidental mutation of the original input.
    return {
        "id": task_identifier,
        "content": deepcopy(task_content),
    }


def enqueue_task_payload(queue: Queue, task: Dict[str, Any]) -> bool:
    """
    Attempts to enqueue the task payload into the provided queue without
    blocking.

    Args:
        queue: The queue to which the task will be added.
        task: The task payload.

    Returns:
        True if enqueued successfully, False otherwise.
    """
    task_id = task.get("id")
    try:
        queue.put(task, block=False)
        # Log at INFO level as requested, including identifier and content.
        logger.info(
            "Task enqueued successfully: id=%s content=%r",
            task_id,
            task.get("content"),
        )
        return True
    except Full:
        logger.warning("Failed to enqueue task (queue full): id=%s", task_id)
        return False
    except Exception as exc:  # Defensive: capture unexpected errors.
        logger.exception(
            "Unexpected error when enqueuing task id=%s: %s", task_id, exc
        )
        return False


def insert_task_to_queue(task_identifier: str, task_content: Dict[str, Any]) -> bool:
    """
    Appends a new task to the global queue for subsequent processing.

    Pipeline:
        1) Validate inputs.
        2) Construct task payload.
        3) Enqueue payload to global task_queue.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and
            parameters.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    # Validation stage
    if not _validate_task_identifier(task_identifier):
        logger.warning(
            "Validation failed for task identifier: %r", task_identifier
        )
        return False

    if not _validate_task_content(task_content):
        logger.warning(
            "Validation failed for task content (expected dict): got %r",
            type(task_content),
        )
        return False

    # Construction stage
    task = create_task_payload(task_identifier, task_content)
    logger.debug("Constructed task payload: %r", task)

    # Enqueue stage
    return enqueue_task_payload(task_queue, task)
