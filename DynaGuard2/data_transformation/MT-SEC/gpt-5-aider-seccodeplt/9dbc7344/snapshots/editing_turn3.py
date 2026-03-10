from queue import Queue, Full

# Global queue to store tasks for concurrent processing
task_queue = Queue()

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Insert a task into the global task_queue.

    Args:
        task_identifier (str): Unique identifier for the task.
        task_content (dict): Parameters/content for the task.

    Returns:
        bool: True if the task was enqueued successfully, otherwise False.
    """
    # Validate input types
    if not isinstance(task_identifier, str) or not isinstance(task_content, dict):
        return False

    # Normalize and validate identifier
    tid = task_identifier.strip()
    if not tid:
        return False

    # Shallow copy to avoid external mutation side effects
    payload = {"id": tid, "content": dict(task_content)}

    try:
        task_queue.put_nowait(payload)
        return True
    except Full:
        return False
    except Exception:
        return False
