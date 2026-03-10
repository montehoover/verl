tasks = []

def add_task(task_id: str, task_description: str, task_info: dict) -> bool:
    """
    Adds a task to the global list of tasks.

    Args:
        task_id: The ID of the task.
        task_description: The description of the task.
        task_info: A dictionary containing additional task details like 'priority' and 'deadline'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_id, str) or \
       not isinstance(task_description, str) or \
       not isinstance(task_info, dict):
        return False
    
    # For simplicity, we'll assume task_id should be unique.
    # If you want to allow duplicate task_ids, remove this check.
    for task in tasks:
        if task['id'] == task_id:
            # Task ID already exists, so not adding.
            # Depending on requirements, you might want to update or handle this differently.
            return False
            
    task_data = {'id': task_id, 'description': task_description}
    task_data.update(task_info)
    tasks.append(task_data)
    return True
