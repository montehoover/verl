from typing import Dict, Any

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
