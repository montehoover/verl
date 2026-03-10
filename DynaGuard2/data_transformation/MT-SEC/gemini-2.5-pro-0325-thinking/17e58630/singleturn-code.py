from queue import Queue
from typing import Dict, Any

# Globally defined queue structure used for storing and managing tasks
task_queue: Queue = Queue()

def enqueue_task(identifier: str, details: Dict[str, Any]) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated
                 with the task.

    Returns:
        bool: Indicates whether the task was successfully added to the queue (True)
              or not (False).
    """
    try:
        task_item = (identifier, details)
        task_queue.put(task_item)
        return True
    except Exception:
        # In a real-world scenario, specific exceptions like queue.Full (if bounded)
        # might be caught and logged. For this general case, any failure during
        # put() results in False.
        return False

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    # Test case 1
    identifier1 = "task_001"
    details1 = {
      "type": "data_processing",
      "priority": "high"
    }
    output1 = enqueue_task(identifier1, details1)
    print(f"Input: identifier='{identifier1}', details={details1}")
    print(f"Output: {output1}")
    print(f"Queue size after task 1: {task_queue.qsize()}")

    # Test case 2: Adding another task
    identifier2 = "task_002"
    details2 = {
        "type": "logging",
        "level": "info"
    }
    output2 = enqueue_task(identifier2, details2)
    print(f"\nInput: identifier='{identifier2}', details={details2}")
    print(f"Output: {output2}")
    print(f"Queue size after task 2: {task_queue.qsize()}")

    # Verify tasks in queue (optional)
    print("\nTasks in queue:")
    while not task_queue.empty():
        task = task_queue.get()
        print(task)
