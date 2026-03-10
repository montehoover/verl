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

from queue import Queue

task_queue = Queue()

def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Adds a task to a global queue for concurrent processing.

    Args:
        identifier: A unique string identifier for the task.
        details: A dictionary containing the parameters for the task.

    Returns:
        True if the task was successfully enqueued, False otherwise.
    """
    if not isinstance(identifier, str) or not isinstance(details, dict):
        return False
    
    try:
        task_data = {"id": identifier, "details": details}
        task_queue.put(task_data)
        return True
    except Exception:
        # In a real-world scenario, log the exception
        return False
