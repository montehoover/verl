from typing import Dict, Any

task_id_list = []

def create_task(task_identifier: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': task_identifier,
        **task_details
    }

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = f"task_{len(task_id_list) + 1}"
    task_id_list.append(task_id)
    task['task_id'] = task_id
    return task
