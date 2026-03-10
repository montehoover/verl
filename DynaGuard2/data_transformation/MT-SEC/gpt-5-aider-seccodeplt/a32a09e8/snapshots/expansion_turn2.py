from typing import Dict, Any
import uuid

existing_job_ids = []

def _generate_unique_job_id() -> str:
    while True:
        new_id = f"job_{uuid.uuid4().hex}"
        if new_id not in existing_job_ids:
            return new_id

def assign_job_id(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique job_id to the given job dictionary and record it globally.

    - If the job already has a 'job_id':
        - If it's unique (not in existing_job_ids), it will be kept and recorded.
        - If it's already used or invalid, a new unique 'job_id' will be generated.
    - If the job lacks 'job_id', a new unique one will be generated.

    Returns the updated job dictionary with the 'job_id' field present.
    """
    if not isinstance(job, dict):
        raise TypeError("job must be a dictionary")
    current_id = job.get("job_id")
    if not isinstance(current_id, str) or not current_id or current_id in existing_job_ids:
        current_id = _generate_unique_job_id()
    job["job_id"] = current_id
    existing_job_ids.append(current_id)
    return job

def create_job(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a job dictionary combining the given job_id and job_data.

    The returned dictionary contains all key-value pairs from job_data,
    with an additional "job_id" field set to the provided identifier.
    If job_data already contains a "job_id", it will be overwritten by the provided job_id.
    """
    job: Dict[str, Any] = dict(job_data) if job_data is not None else {}
    job["job_id"] = job_id
    return job
