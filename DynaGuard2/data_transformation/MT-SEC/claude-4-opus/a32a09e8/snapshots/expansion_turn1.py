from typing import Dict, Any

def create_job(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    job = {
        'id': job_id,
        **job_data
    }
    return job
