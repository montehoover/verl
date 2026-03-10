from queue import Queue, Full

task_queue = Queue()


def create_task(uid: str, data: dict) -> dict:
    """
    Pure function to create a task payload from inputs.

    Args:
        uid (str): Unique identifier for the task.
        data (dict): Task details and parameters.

    Returns:
        dict: The task payload.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a non-empty string")
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    return {"uid": uid, "data": data}


def enqueue_task(task: dict) -> bool:
    """
    Side-effectful function that enqueues a task into the global queue.

    Args:
        task (dict): The task payload to enqueue.

    Returns:
        bool: True if enqueued successfully; False otherwise.
    """
    try:
        task_queue.put(task, block=False)
        return True
    except Full:
        return False
    except Exception:
        return False


def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    Integrates new tasks into a global queue for future execution.

    Args:
        uid (str): A unique string that serves as an identifier for each task.
        data (dict): A dictionary containing the details and parameters associated with the task.

    Returns:
        bool: True if the task was successfully added to the queue; False otherwise.
    """
    try:
        task = create_task(uid, data)
    except Exception:
        return False

    return enqueue_task(task)
