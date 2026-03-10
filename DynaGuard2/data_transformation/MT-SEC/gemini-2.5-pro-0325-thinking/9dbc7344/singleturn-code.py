from queue import Queue, Full

task_queue = Queue()

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Appends a new task to the global task queue.

    Args:
        task_identifier: A unique string identifier for the task.
        task_content: A dictionary containing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    task_item = (task_identifier, task_content)
    try:
        task_queue.put_nowait(task_item)
        return True
    except Full:
        # This exception occurs if put_nowait is called on a bounded queue that is full.
        # For the default unbounded Queue, this path will not be taken.
        return False
