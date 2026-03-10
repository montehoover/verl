from typing import Dict, Any

def create_task(task_identifier: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': task_identifier,
        **task_details
    }
