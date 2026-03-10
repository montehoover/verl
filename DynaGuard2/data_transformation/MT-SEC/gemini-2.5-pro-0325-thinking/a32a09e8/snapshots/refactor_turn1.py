from queue import Queue, Full

# task_queue: A globally accessible Queue object used for task management and storage
task_queue = Queue()

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.

    This function is tasked with incorporating new tasks into a global queue
    for subsequent execution.

    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict): A dictionary encompassing the task's details, parameters,
                         and other relevant information.

    Returns:
        bool: A boolean value indicating success (True) or failure (False) of
              task addition to the queue.
    """
    try:
        # Using put_nowait for a non-blocking call.
        # If task_queue were initialized with a maxsize and became full,
        # this would raise queue.Full. For an unbounded Queue(), Full is not raised.
        task_queue.put_nowait((job_id, job_data))
        return True
    except Full:
        # This handles the case where the queue might be full (if bounded).
        return False
    except Exception:
        # Catches any other unexpected error during the put operation.
        # In a production system, it would be good practice to log this error.
        return False
