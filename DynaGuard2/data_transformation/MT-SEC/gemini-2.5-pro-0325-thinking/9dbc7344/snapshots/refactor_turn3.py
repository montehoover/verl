import logging
from queue import Queue
from typing import Tuple

# Configure basic logging
# This will output logs to the console with a timestamp, log level, and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

task_queue = Queue()  # Global queue for task storage

def _create_task_item(task_identifier: str, task_content: dict) -> Tuple[str, dict]:
    """
    Creates a task item tuple from the identifier and content.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        A tuple containing the task_identifier and task_content.
    """
    return task_identifier, task_content  # Parentheses are optional for returning a tuple

def _enqueue_task_item(queue: Queue, task_item: Tuple[str, dict]) -> bool:
    """
    Enqueues a given task item into the provided queue and logs the operation.

    Args:
        queue: The queue to which the task item will be added.
        task_item: The task item (tuple) to be enqueued.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    try:
        queue.put(task_item)
        logging.info(
            f"Task enqueued successfully: ID='{task_item[0]}', Content='{task_item[1]}'"
        )
        return True
    except Exception as e:
        # Log the error if enqueuing fails.
        # queue.Full is a common exception if the queue has a maximum size.
        logging.error(
            f"Failed to enqueue task: ID='{task_item[0]}', Content='{task_item[1]}'. Error: {e}"
        )
        return False

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Creates a task item and appends it to the global task queue.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    task_item = _create_task_item(task_identifier, task_content)
    return _enqueue_task_item(task_queue, task_item)
