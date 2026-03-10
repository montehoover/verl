from queue import Queue

task_queue = Queue()


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

    # Validate inputs
    if not isinstance(job_id, str) or not job_id.strip():
        return False
    if not isinstance(job_data, dict):
        return False

    task = {"job_id": job_id, "job_data": job_data}

    try:
        # Non-blocking insert to avoid waiting in parallel contexts
        task_queue.put_nowait(task)
        return True
    except Exception:
        # Catch any unexpected failure (e.g., if a bounded queue is full)
        return False
