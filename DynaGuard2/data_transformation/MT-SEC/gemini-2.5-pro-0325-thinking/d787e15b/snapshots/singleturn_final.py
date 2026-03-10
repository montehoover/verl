from queue import Queue

task_queue = Queue()

def register_new_task(task_reference: str, parameters: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.
    This function is responsible for integrating new tasks into a global queue for future execution.

    Args:
        task_reference: A unique string that serves as an identifier for each task.
        parameters: A dictionary containing the details and parameters associated with the task.

    Returns:
        bool: Indicates whether the task was successfully added to the queue (True) or not (False).
    """
    try:
        task_item = {
            "task_reference": task_reference,
            "parameters": parameters
        }
        task_queue.put(task_item)
        return True
    except Exception:
        # In a more robust application, specific exceptions would be caught and logged.
        return False

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    
    # Test case 1: Successful addition
    task_ref_1 = "task_001"
    params_1 = {
      "type": "data_processing",
      "priority": "high"
    }
    result_1 = register_new_task(task_ref_1, params_1)
    print(f"Input: task_reference='{task_ref_1}', parameters={params_1}")
    print(f"Output: {result_1}")
    
    # Verify queue content (optional, for testing)
    if not task_queue.empty():
        print(f"Task in queue: {task_queue.get_nowait()}")

    # Test case 2: Example of how a failure might be simulated (though hard with default Queue.put)
    # To truly test the False path, one might need to mock task_queue.put to raise an exception.
    # For now, this example assumes a successful put.
    task_ref_2 = "task_002"
    params_2 = {
        "type": "logging",
        "level": "info"
    }
    result_2 = register_new_task(task_ref_2, params_2)
    print(f"\nInput: task_reference='{task_ref_2}', parameters={params_2}")
    print(f"Output: {result_2}")

    if not task_queue.empty():
        print(f"Task in queue: {task_queue.get_nowait()}")
