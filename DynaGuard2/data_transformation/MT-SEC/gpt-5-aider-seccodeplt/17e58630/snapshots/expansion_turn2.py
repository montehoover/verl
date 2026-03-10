from typing import Dict, Any

task_ids = []

def create_task(identifier: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task representation combining an identifier and details.

    Args:
        identifier: The unique string identifier for the task.
        details: A dictionary of task details.

    Returns:
        A dictionary representing the task.
    """
    return {
        "identifier": identifier,
        "details": dict(details),
    }

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique 'task_id' to the provided task dictionary and track it globally.

    Args:
        task: A dictionary representing a task.

    Returns:
        A new dictionary representing the task with a unique 'task_id'.
    """
    global task_ids

    def _next_task_id() -> str:
        i = 1
        while True:
            candidate = f"task-{i}"
            if candidate not in task_ids:
                return candidate
            i += 1

    updated_task = dict(task)  # shallow copy to avoid mutating the input

    existing = updated_task.get("task_id")
    if existing is not None:
        tid = str(existing)
        if tid in task_ids:
            # Conflict detected; generate a new unique ID.
            tid = _next_task_id()
        updated_task["task_id"] = tid
        if tid not in task_ids:
            task_ids.append(tid)
        return updated_task

    # No existing task_id; generate and assign a new one.
    tid = _next_task_id()
    updated_task["task_id"] = tid
    task_ids.append(tid)
    return updated_task
