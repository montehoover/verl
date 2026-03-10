from queue import Queue, Full

TASKS = []

# Global task queue for parallel task handling
task_queue = Queue()


def add_task(task_name: str, task_info: str, task_details: dict) -> bool:
    """
    Add a task to the global TASKS list.

    Args:
        task_name (str): The name of the task.
        task_info (str): Additional info or description of the task.
        task_details (dict): Additional details for the task, e.g., {"priority": "high", "deadline": "2025-12-31"}.

    Returns:
        bool: True if the task was added successfully, False otherwise.
    """
    if (
        not isinstance(task_name, str)
        or not isinstance(task_info, str)
        or not isinstance(task_details, dict)
    ):
        return False

    name = task_name.strip()
    info = task_info.strip()

    if not name or not info:
        return False

    # Store a shallow copy of details to avoid external mutations affecting stored task
    TASKS.append({"task_name": name, "task_info": info, "task_details": dict(task_details)})
    return True


def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Insert a task into the global task_queue for parallel processing.

    Args:
        job_id (str): Unique identifier for the task.
        job_data (dict): Task details payload.

    Returns:
        bool: True if the task was enqueued successfully, False otherwise.
    """
    if not isinstance(job_id, str) or not isinstance(job_data, dict):
        return False

    jid = job_id.strip()
    if not jid:
        return False

    payload = {"job_id": jid, "job_data": dict(job_data)}

    try:
        task_queue.put_nowait(payload)
        return True
    except Full:
        return False
    except Exception:
        return False
