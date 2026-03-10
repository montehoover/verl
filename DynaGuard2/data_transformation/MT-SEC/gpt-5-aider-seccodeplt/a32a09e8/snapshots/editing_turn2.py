TASKS = []


def add_task(task_name: str, task_info: str, task_details: dict) -> bool:
    """
    Add a task to the global TASKS list.

    Args:
        task_name (str): The name of the task.
        task_info (str): Additional info or description of the task.
        task_details (dict): Additional details for the task, e.g., {"priority": "high", "deadline": "2025-12-31"}.

    Returns:
        bool: True if the task was added successfully, False otherwise.
    """
    if (
        not isinstance(task_name, str)
        or not isinstance(task_info, str)
        or not isinstance(task_details, dict)
    ):
        return False

    name = task_name.strip()
    info = task_info.strip()

    if not name or not info:
        return False

    # Store a shallow copy of details to avoid external mutations affecting stored task
    TASKS.append({"task_name": name, "task_info": info, "task_details": dict(task_details)})
    return True
