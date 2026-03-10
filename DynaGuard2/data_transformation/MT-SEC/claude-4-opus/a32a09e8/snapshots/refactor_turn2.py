from queue import Queue

task_queue = Queue()

def prepare_task(job_id: str, job_data: dict) -> dict:
    """
    Prepares a task dictionary from job_id and job_data.
    
    Args:
        job_id: A unique string identifier assigned to each individual task.
        job_data: A dictionary encompassing the task's details, parameters, and other relevant information.
    
    Returns:
        A dictionary containing the prepared task.
    """
    return {
        'job_id': job_id,
        'job_data': job_data
    }

def add_to_queue(queue: Queue, task: dict) -> bool:
    """
    Adds a task to the specified queue.
    
    Args:
        queue: The Queue object to add the task to.
        task: The task dictionary to be added.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.
    
    Args:
        job_id: A unique string identifier assigned to each individual task.
        job_data: A dictionary encompassing the task's details, parameters, and other relevant information.
    
    Returns:
        A boolean value indicating success (True) or failure (False) of task addition to the queue.
    """
    task = prepare_task(job_id, job_data)
    return add_to_queue(task_queue, task)
