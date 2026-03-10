from queue import Queue, Full

# Global list to store tasks
tasks = []

# Global queue to store tasks for concurrent processing
task_queue = Queue()

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

def enqueue_task(identifier, details):
    """
    Enqueue a task into the global task_queue.
    identifier: str - unique task ID
    details: dict - task parameters
    Returns True if enqueued successfully, else False.
    """
    # Validate identifier
    if not isinstance(identifier, str):
        return False
    task_id = identifier.strip()
    if not task_id:
        return False

    # Validate details
    if not isinstance(details, dict):
        return False

    # Prepare task payload (copy details to avoid external mutation)
    task_payload = {
        'id': task_id,
        'details': dict(details)
    }

    # Non-blocking enqueue
    try:
        task_queue.put_nowait(task_payload)
        return True
    except Full:
        return False
    except Exception:
        return False
