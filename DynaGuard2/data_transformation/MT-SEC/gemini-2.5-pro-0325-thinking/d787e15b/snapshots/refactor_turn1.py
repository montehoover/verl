from queue import Queue

task_queue = Queue()

def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated
                    with the task.

    Returns:
        Indicates whether the task was successfully added to the queue (True)
        or not (False).
    """
    try:
        task_queue.put((task_reference, parameters))
        return True
    except Exception:
        # In a real-world scenario, specific exceptions should be caught and logged.
        # For instance, if the queue could be full (e.g., a bounded queue with block=False).
        # However, a standard Queue.put() is blocking and won't raise Full unless
        # it's a Full exception from a bounded queue with non-blocking put.
        # For simplicity here, catching a generic Exception.
        return False
