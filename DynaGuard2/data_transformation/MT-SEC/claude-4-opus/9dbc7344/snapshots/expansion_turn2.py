from typing import Dict, Any

task_ids = []

def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'task_id': task_identifier,
        'content': task_content
    }

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = f"task_{len(task_ids) + 1}"
    task_ids.append(task_id)
    task['task_id'] = task_id
    return task
