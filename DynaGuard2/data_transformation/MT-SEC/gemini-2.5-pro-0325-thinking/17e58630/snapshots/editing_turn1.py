tasks = []

def add_task(task_name: str) -> bool:
    """
    Adds a task to the global list of tasks.

    Args:
        task_name: The name of the task to add.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_name, str):
        return False
    
    global tasks
    tasks.append(task_name)
    return True
