from queue import Queue

task_queue = Queue()

def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.

    This function is tasked with incorporating new tasks into a global queue
    for subsequent execution.

    Args:
        job_id: A unique string identifier assigned to each individual task.
        job_data: A dictionary encompassing the task's details, parameters,
                  and other relevant information.

    Returns:
        A boolean value indicating success (True) or failure (False) of task
        addition to the queue.
    """
    try:
        task_queue.put((job_id, job_data))
        return True
    except Exception:
        # In a real-world scenario, you might want to log the exception here
        return False

if __name__ == '__main__':
    # Example Usage based on the provided input/output
    
    # Test case 1: Successful insertion
    job_id_1 = "job_001"
    job_data_1 = {
      "type": "data_processing",
      "priority": "high"
    }
    result_1 = insert_task_to_queue(job_id_1, job_data_1)
    print(f"Input: job_id='{job_id_1}', job_data={job_data_1}")
    print(f"Output: {result_1}")
    print(f"Expected: True")
    print(f"Queue size after insertion: {task_queue.qsize()}")
    if not task_queue.empty():
        print(f"Task in queue: {task_queue.get_nowait()}")
    print("-" * 20)

    # Test case 2: Another successful insertion
    job_id_2 = "job_002"
    job_data_2 = {
        "type": "email_notification",
        "recipient": "user@example.com"
    }
    result_2 = insert_task_to_queue(job_id_2, job_data_2)
    print(f"Input: job_id='{job_id_2}', job_data={job_data_2}")
    print(f"Output: {result_2}")
    print(f"Expected: True")
    print(f"Queue size after insertion: {task_queue.qsize()}")
    if not task_queue.empty():
        print(f"Task in queue: {task_queue.get_nowait()}") # Will get job_002 as job_001 was retrieved
    print("-" * 20)

    # To demonstrate failure, we would need to mock queue.put() to raise an exception,
    # as standard Queue.put() is blocking by default and doesn't easily fail
    # unless a specific Queue type (e.g. with maxsize and non-blocking put) is used
    # and it's full. For this example, we assume it works or an unexpected error occurs.

    # Verify remaining items in queue if any
    print("Remaining items in queue:")
    while not task_queue.empty():
        print(task_queue.get_nowait())
