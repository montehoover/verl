from queue import Queue

task_queue = Queue()

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    try:
        task_item = (task_identifier, task_content)
        task_queue.put(task_item)
        return True
    except Exception:
        # Catching a broad exception here as queue.Full might be relevant
        # for bounded queues, or other unforeseen issues during enqueuing.
        return False
