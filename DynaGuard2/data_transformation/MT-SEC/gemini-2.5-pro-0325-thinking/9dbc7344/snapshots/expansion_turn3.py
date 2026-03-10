from typing import Dict, Any, List
from queue import Queue

task_ids: List[str] = []
task_queue: Queue = Queue()

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task_id to the task and records it.

    Args:
        task: The task dictionary.

    Returns:
        The task dictionary updated with a 'task_id'.
    """
    new_id = f"task_{len(task_ids) + 1}"
    task_ids.append(new_id)
    task['task_id'] = new_id
    return task

def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encapsulates task details.

    Args:
        task_identifier: The unique identifier for the task.
        task_content: A dictionary containing the specifics of the task.

    Returns:
        A dictionary combining the task identifier and its content.
    """
    return {
        "task_id": task_identifier,
        "content": task_content
    }

def insert_task_to_queue(task_identifier: str, task_content: Dict[str, Any]) -> bool:
    """
    Creates a task and adds it to the global task queue.

    Args:
        task_identifier: The unique identifier for the task.
        task_content: A dictionary containing the specifics of the task.

    Returns:
        True if the task was successfully enqueued, False otherwise.
    """
    try:
        task = create_task(task_identifier, task_content)
        task_queue.put(task)
        return True
    except Exception:
        # In a real-world scenario, specific exceptions should be caught and logged.
        return False
