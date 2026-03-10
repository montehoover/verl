from queue import Queue
from typing import Any, Tuple

task_queue = Queue()


def _prepare_task_data(task_reference: str, parameters: dict) -> Tuple[str, dict]:
    """
    Prepares the task data for queueing.

    Args:
        task_reference: The unique identifier for the task.
        parameters: The parameters associated with the task.

    Returns:
        A tuple containing the task reference and parameters.
    """
    return (task_reference, parameters)


def _add_task_to_queue(queue: Queue, task_data: Any) -> bool:
    """
    Adds the prepared task data to the specified queue.

    Args:
        queue: The queue to add the task to.
        task_data: The data representing the task.

    Returns:
        True if the task was successfully added, False otherwise.
    """
    try:
        queue.put(task_data)
        return True
    except Exception:
        # In a real-world scenario, specific exceptions should be caught and logged.
        return False


def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated
                    with the task.

    Returns:
        Indicates whether the task was successfully added to the queue (True)
        or not (False).
    """
    task_data = _prepare_task_data(task_reference, parameters)
    return _add_task_to_queue(task_queue, task_data)
