tasks = []

def add_task(task_name: str, task_info: dict = None) -> bool:
    """
    Adds a task with optional details to the global list of tasks.

    Args:
        task_name: The name of the task to add.
        task_info: An optional dictionary with task details like 'priority' and 'due_date'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_name, str):
        return False
    if task_info is not None and not isinstance(task_info, dict):
        return False
    
    global tasks
    task_entry = {"name": task_name}
    if task_info:
        task_entry.update(task_info)
    tasks.append(task_entry)
    return True
