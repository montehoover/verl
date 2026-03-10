# Global list to store tasks
tasks = []

def add_task(task_id: str, task_description: str) -> bool:
    """
    Add a task to the global tasks list.

    Args:
        task_id (str): Unique identifier for the task.
        task_description (str): Description of the task.

    Returns:
        bool: True if the task was added successfully, otherwise False.
    """
    # Validate input types
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False

    # Normalize and validate content
    tid = task_id.strip()
    desc = task_description.strip()
    if not tid or not desc:
        return False

    # Prevent duplicate task IDs
    for task in tasks:
        if isinstance(task, dict) and task.get("id") == tid:
            return False

    # Add the task
    tasks.append({"id": tid, "description": desc})
    return True
