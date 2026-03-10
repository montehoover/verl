from queue import Queue


task_queue = Queue()


def prepare_task(job_id: str, job_data: dict) -> dict:
    """
    Prepares a task dictionary from job_id and job_data.
    
    This function takes a job identifier and its associated data and combines
    them into a standardized task dictionary format that can be processed by
    the queue system.
    
    Args:
        job_id: A unique string identifier assigned to each individual task.
            This ID is used to track and reference the task throughout its
            lifecycle in the processing system.
        job_data: A dictionary encompassing the task's details, parameters,
            and other relevant information needed for task execution.
    
    Returns:
        dict: A dictionary containing the prepared task with the following structure:
            {
                'job_id': str,     # The unique task identifier
                'job_data': dict   # The task's data and parameters
            }
    """
    return {
        'job_id': job_id,
        'job_data': job_data
    }


def add_to_queue(queue: Queue, task: dict) -> bool:
    """
    Adds a task to the specified queue.
    
    This function attempts to add a prepared task dictionary to a Queue object.
    It handles any potential exceptions that might occur during the queue
    insertion process and returns a boolean indicating success or failure.
    
    Args:
        queue: The Queue object to add the task to. This should be a
            thread-safe queue instance from the queue module.
        task: The task dictionary to be added. This should be a properly
            formatted task dictionary containing job_id and job_data.
    
    Returns:
        bool: True if the task was successfully added to the queue,
            False if any exception occurred during the insertion process.
    """
    try:
        queue.put(task)
        return True
    except Exception:
        return False


def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.
    
    This is the main entry point for adding new tasks to the global task queue.
    It coordinates the task preparation and queue insertion process by utilizing
    helper functions to maintain separation of concerns and improve code
    maintainability.
    
    The function first prepares the task data into the appropriate format, then
    attempts to add it to the global task queue. This two-step process ensures
    that tasks are properly formatted before being queued for processing.
    
    Args:
        job_id: A unique string identifier assigned to each individual task.
            This identifier should be unique across all tasks in the system
            to ensure proper task tracking and management.
        job_data: A dictionary encompassing the task's details, parameters,
            and other relevant information. The structure and content of this
            dictionary depend on the specific requirements of the task being
            performed.
    
    Returns:
        bool: A boolean value indicating success (True) or failure (False)
            of task addition to the queue. Returns True if the task was
            successfully prepared and added to the queue, False otherwise.
    """
    task = prepare_task(job_id, job_data)
    return add_to_queue(task_queue, task)
