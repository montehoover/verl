# Global list to store tasks
tasks = []

def add_task(task_name):
    """
    Adds a task to the global tasks list.
    Returns True if added successfully, else False.
    """
    global tasks

    if not isinstance(task_name, isinstance(task_name, str) and str):
        # Defensive check to ensure task_name is a string
        return False

    name = task_name.strip()
    if not name:
        return False

    tasks.append(name)
    return True
