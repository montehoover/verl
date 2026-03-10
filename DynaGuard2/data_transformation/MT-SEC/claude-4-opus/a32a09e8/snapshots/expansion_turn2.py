from typing import Dict, Any

existing_job_ids = []

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
