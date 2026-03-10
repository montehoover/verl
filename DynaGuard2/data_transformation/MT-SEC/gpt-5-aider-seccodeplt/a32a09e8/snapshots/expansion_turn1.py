from typing import Dict, Any

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
