from typing import Dict, Any, List
import uuid
from queue import Queue

existing_job_ids: List[str] = []
task_queue: Queue = Queue()

def create_job(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a job dictionary by combining a job_id and job_data.

    Args:
        job_id: The unique identifier for the job.
        job_data: A dictionary containing the details of the job.

    Returns:
        A dictionary representing the job, including its id and data.
    """
    job = {"id": job_id}
    job.update(job_data)
    return job

def assign_job_id(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique job_id to the job_data and records it.

    Args:
        job_data: A dictionary containing the details of the job.

    Returns:
        The job_data dictionary updated with a 'job_id'.
    """
    # Generate a unique job_id (simple example, could be more robust)
    # For this example, we'll use a simple counter based on existing IDs,
    # but a UUID is generally better for true uniqueness.
    # Let's use a UUID to ensure uniqueness and avoid collisions.
    new_id = str(uuid.uuid4())
    
    while new_id in existing_job_ids: # Ensure uniqueness, though highly unlikely with UUIDs
        new_id = str(uuid.uuid4())
        
    job_data['job_id'] = new_id
    existing_job_ids.append(new_id)
    return job_data

def insert_task_to_queue(job_id: str, job_data: Dict[str, Any]) -> bool:
    """
    Creates a job and inserts it into the global task queue.

    Args:
        job_id: The unique identifier for the job.
        job_data: A dictionary containing the details of the job.

    Returns:
        True if the task was successfully added to the queue, False otherwise.
        (In this implementation, it always returns True as put() is blocking
         or raises an exception on a full queue if non-blocking, which is not handled here).
    """
    job = create_job(job_id, job_data)
    try:
        task_queue.put(job)
        return True
    except Exception:
        # Basic error handling, could be more specific depending on Queue type and usage
        # For a standard queue.Queue, put() can block if the queue is full and has a maxsize.
        # If maxsize is 0 (default, unbounded), put() will not block.
        # If an error occurs during put (e.g. if the queue is full and non-blocking),
        # it might raise Full. For simplicity, we catch a generic Exception.
        return False
