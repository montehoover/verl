from typing import Dict, Any
from queue import Queue

existing_job_ids = []
task_queue = Queue()

def create_job(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    job = {
        'id': job_id,
        **job_data
    }
    return job

def assign_job_id(job: Dict[str, Any]) -> Dict[str, Any]:
    import uuid
    job_id = str(uuid.uuid4())
    existing_job_ids.append(job_id)
    job['job_id'] = job_id
    return job

def insert_task_to_queue(job_id: str, job_data: Dict[str, Any]) -> bool:
    try:
        job = create_job(job_id, job_data)
        task_queue.put(job)
        return True
    except Exception:
        return False
