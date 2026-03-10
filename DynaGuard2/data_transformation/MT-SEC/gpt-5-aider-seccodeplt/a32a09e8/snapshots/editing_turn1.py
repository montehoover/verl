TASKS = []


def add_task(task_name: str, task_info: str) -> bool:
    """
    Add a task to the global TASKS list.

    Args:
        task_name (str): The name of the task.
        task_info (str): Additional info or description of the task.

    Returns:
        bool: True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_name, str) or not isinstance(task_info, str):
        return False

    name = task_name.strip()
    info = task_info.strip()

    if not name or not info:
        return False

    TASKS.append({"task_name": name, "task_info": info})
    return True
