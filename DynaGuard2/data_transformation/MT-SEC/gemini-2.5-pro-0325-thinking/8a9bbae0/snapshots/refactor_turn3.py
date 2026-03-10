import logging
from queue import Queue
from typing import Any, Dict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

task_queue = Queue()


def _validate_task_inputs(task_identifier: str, task_details: dict) -> bool:
    """
    Validates the inputs for a task.

    Args:
        task_identifier: The identifier for the task.
        task_details: The details of the task.

    Returns:
        True if inputs are valid, False otherwise.
    """
    if not isinstance(task_identifier, str) or not task_identifier:
        logging.warning(f"Validation failed for task_identifier: Must be a non-empty string. Received: {task_identifier}")
        return False
    if not isinstance(task_details, dict):
        logging.warning(f"Validation failed for task_details for task '{task_identifier}': Must be a dictionary. Received type: {type(task_details)}")
        return False
    return True


def _prepare_task_for_queue(task_identifier: str, task_details: dict) -> Dict[str, Any]:
    """
    Prepares the task payload for queueing.

    Args:
        task_identifier: The identifier for the task.
        task_details: The details of the task.

    Returns:
        A dictionary representing the task to be enqueued.
    """
    return {
        "identifier": task_identifier,
        "details": task_details
    }


def _attempt_queue_task(queue_instance: Queue, task_payload: Dict[str, Any]) -> bool:
    """
    Attempts to add the prepared task payload to the given queue.

    Args:
        queue_instance: The queue to add the task to.
        task_payload: The task payload to enqueue.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    task_id = task_payload.get("identifier", "Unknown task")
    try:
        queue_instance.put(task_payload)
        logging.info(f"Task '{task_id}' successfully added to the queue.")
        return True
    except Exception as e:
        logging.error(f"Failed to add task '{task_id}' to queue. Error: {e}", exc_info=True)
        return False


def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    if not _validate_task_inputs(task_identifier, task_details):
        # Logging for validation failure is handled within _validate_task_inputs
        return False

    task_payload = _prepare_task_for_queue(task_identifier, task_details)

    if _attempt_queue_task(task_queue, task_payload):
        # Successful enqueue logging is handled within _attempt_queue_task
        return True
    else:
        # Error logging during enqueue attempt is handled within _attempt_queue_task
        logging.error(f"Enqueueing task '{task_identifier}' failed overall.")
        return False
