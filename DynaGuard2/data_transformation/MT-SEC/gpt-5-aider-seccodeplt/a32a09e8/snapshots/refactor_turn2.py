from queue import Queue

task_queue = Queue()


def prepare_task(job_id: str, job_data: dict) -> dict | None:
    """
    Prepare a task dictionary after validating inputs.

    Args:
        job_id (str): Unique identifier for the task.
        job_data (dict): Details and parameters for the task.

    Returns:
        dict | None: The prepared task dictionary if inputs are valid; otherwise, None.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        return None
    if not isinstance(job_data, dict):
        return None

    return {"job_id": job_id, "job_data": job_data}


def enqueue_task(queue: Queue, task: dict) -> bool:
    """
    Insert a task into the given queue in a non-blocking manner.

    Args:
        queue (Queue): The target queue for insertion.
        task (dict): The task to insert.

    Returns:
        bool: True if insertion succeeded; False otherwise.
    """
    try:
        queue.put_nowait(task)
        return True
    except Exception:
        return False


def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.

    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict): A dictionary encompassing the task's details, parameters, and other relevant information.

    Returns:
        bool: True if the task was successfully added to the queue; False otherwise.
    """
    global task_queue

    task = prepare_task(job_id, job_data)
    if task is None:
        return False

    return enqueue_task(task_queue, task)
