from typing import Dict, Any
from queue import Queue

task_ids = []
task_queue = Queue()

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

def insert_task_to_queue(task_identifier: str, task_content: Dict[str, Any]) -> bool:
    try:
        task = create_task(task_identifier, task_content)
        task_queue.put(task)
        return True
    except:
        return False
