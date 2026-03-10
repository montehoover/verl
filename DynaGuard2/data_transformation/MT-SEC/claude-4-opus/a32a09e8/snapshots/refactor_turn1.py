from queue import Queue

task_queue = Queue()

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.
    
    Args:
        job_id: A unique string identifier assigned to each individual task.
        job_data: A dictionary encompassing the task's details, parameters, and other relevant information.
    
    Returns:
        A boolean value indicating success (True) or failure (False) of task addition to the queue.
    """
    try:
        task = {
            'job_id': job_id,
            'job_data': job_data
        }
        task_queue.put(task)
        return True
    except Exception:
        return False
