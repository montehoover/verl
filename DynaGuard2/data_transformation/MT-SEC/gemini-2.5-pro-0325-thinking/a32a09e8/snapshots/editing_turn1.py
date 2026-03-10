tasks = []

def add_task(task_name: str, task_info: str) -> bool:
    """
    Adds a task to the global tasks list.

    Args:
        task_name: The name of the task.
        task_info: A string containing information about the task.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_name, str) or not isinstance(task_info, str):
        return False
    
    tasks.append({"name": task_name, "info": task_info})
    return True
