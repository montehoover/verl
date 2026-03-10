from typing import Dict, Any

def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'task_id': task_identifier,
        'content': task_content
    }
