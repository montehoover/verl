# Global list to store tasks
tasks = []

def add_task(task_id: str, task_description: str, task_metadata: dict) -> bool:
    """
    Add a task to the global tasks list.

    Args:
        task_id (str): Unique identifier for the task.
        task_description (str): Description of the task.
        task_metadata (dict): Additional metadata for the task (e.g., priority, deadline).

    Returns:
        bool: True if the task was added successfully, otherwise False.
    """
    # Validate input types
    if not isinstance(task_id, str) or not isinstance(task_description, str) or not isinstance(task_metadata, dict):
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

    # Add the task with metadata
    # Copy metadata to avoid external mutations affecting stored task
    metadata = dict(task_metadata)

    tasks.append({"id": tid, "description": desc, "metadata": metadata})
    return True
