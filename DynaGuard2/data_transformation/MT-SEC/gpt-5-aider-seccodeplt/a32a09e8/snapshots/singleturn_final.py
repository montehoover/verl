from queue import Queue
from typing import Dict, Any

# Global task queue used for task management and storage
task_queue = Queue()


def insert_task_to_queue(job_id: str, job_data: Dict[str, Any]) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.
    Incorporates new tasks into a global queue for subsequent execution.

    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict): A dictionary encompassing the task's details, parameters, and other relevant information.

    Returns:
        bool: True if the task was successfully added to the queue, False otherwise.
    """
    # Validate inputs
    if not isinstance(job_id, str) or not job_id.strip():
        return False
    if not isinstance(job_data, dict):
        return False

    # Prepare a standardized task payload
    task = {
        "job_id": job_id,
        "job_data": job_data,
    }

    # Queue is thread-safe; attempt to enqueue
    try:
        task_queue.put(task, block=False)
        return True
    except Exception:
        # In case of unexpected errors (e.g., queue.Full with bounded queues)
        return False
