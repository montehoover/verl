from queue import Queue

tasks = []
task_queue = Queue()

def add_task(task_name: str, task_info: str, task_details: dict) -> bool:
    """
    Adds a task to the global tasks list.

    Args:
        task_name: The name of the task.
        task_info: A string containing information about the task.
        task_details: A dictionary containing additional details like 'priority' and 'deadline'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_name, str) or \
       not isinstance(task_info, str) or \
       not isinstance(task_details, dict):
        return False
    
    tasks.append({"name": task_name, "info": task_info, "details": task_details})
    return True

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Inserts a task into the global task queue.

    Args:
        job_id: The unique identifier for the task.
        job_data: A dictionary containing the task details.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(job_id, str) or not isinstance(job_data, dict):
        return False
    
    try:
        task_queue.put({"job_id": job_id, "data": job_data})
        return True
    except Exception:
        # Optionally log the exception here
        return False
