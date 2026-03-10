from queue import Queue, Full
from typing import Any, Dict, Optional

task_queue = Queue()


def validate_inputs(task_reference: str, parameters: dict) -> bool:
    """
    Validates input arguments for registering a new task.

    Args:
        task_reference (str): A unique string identifier for the task.
        parameters (dict): A dictionary containing details and parameters for the task.

    Returns:
        bool: True if inputs are valid, False otherwise.
    """
    if not isinstance(task_reference, str) or not task_reference.strip():
        return False
    if not isinstance(parameters, dict):
        return False
    return True


def prepare_task_payload(task_reference: str, parameters: dict) -> Dict[str, Any]:
    """
    Prepares the task payload for queueing.

    Args:
        task_reference (str): A unique string identifier for the task.
        parameters (dict): A dictionary containing details and parameters for the task.

    Returns:
        Dict[str, Any]: A dictionary representing the task payload.
    """
    return {
        "task_reference": task_reference,
        "parameters": dict(parameters),  # copy to avoid external mutation
    }


def enqueue_task(queue: Queue, payload: Dict[str, Any], block: bool = False) -> bool:
    """
    Attempts to enqueue the provided payload into the given queue.

    Args:
        queue (Queue): The queue to insert the payload into.
        payload (Dict[str, Any]): The task payload to enqueue.
        block (bool): Whether to block if the queue is full. Defaults to False.

    Returns:
        bool: True if enqueued successfully, False otherwise.
    """
    try:
        queue.put(payload, block=block)
        return True
    except Full:
        return False
    except Exception:
        return False


def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Registers a new task into the global task_queue.

    Args:
        task_reference (str): A unique string identifier for the task.
        parameters (dict): A dictionary containing details and parameters for the task.

    Returns:
        bool: True if the task was successfully added to the queue, False otherwise.
    """
    if not validate_inputs(task_reference, parameters):
        return False

    task_payload = prepare_task_payload(task_reference, parameters)
    return enqueue_task(task_queue, task_payload, block=False)
