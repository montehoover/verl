# Global list to store tasks
tasks = []

def add_task(task_name, task_info=None):
    """
    Adds a task with additional info to the global tasks list.
    Returns True if added successfully, else False.
    """
    global tasks

    # Validate task_name
    if not isinstance(task_name, str):
        return False

    name = task_name.strip()
    if not name:
        return False

    # Handle and validate task_info
    if task_info is None:
        task_info = {}
    elif not isinstance(task_info, dict):
        return False

    # Construct the task entry combining name and details
    task_entry = {'name': name}
    task_entry.update(task_info)

    tasks.append(task_entry)
    return True
