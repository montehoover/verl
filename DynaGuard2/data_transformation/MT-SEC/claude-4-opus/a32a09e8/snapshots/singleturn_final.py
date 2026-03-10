from queue import Queue

task_queue = Queue()

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.
    
    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict): A dictionary encompassing the task's details, parameters, and other relevant information.
    
    Returns:
        bool: A boolean value indicating success (True) or failure (False) of task addition to the queue.
    """
    try:
        # Create a task dictionary containing both the job_id and job_data
        task = {
            'job_id': job_id,
            'job_data': job_data
        }
        
        # Add the task to the global queue
        task_queue.put(task)
        
        # Return True to indicate successful insertion
        return True
    except Exception:
        # Return False if any error occurs during insertion
        return False
