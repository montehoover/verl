from queue import Queue

task_queue = Queue()

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Enqueues a task into the global task_queue.

    Args:
        task_identifier: The unique ID of the task.
        task_details: A dictionary containing task specifics.

    Returns:
        True if the task was enqueued successfully, False otherwise.
    """
    if not isinstance(task_identifier, str) or not isinstance(task_details, dict):
        return False
    
    try:
        # We can store a tuple or a dictionary in the queue.
        # Storing a dictionary might be more descriptive.
        task_data = {'id': task_identifier, 'details': task_details}
        task_queue.put(task_data)
        return True
    except Exception:
        # Handle potential queue errors, though put() on a standard Queue
        # usually blocks or raises Full if maxsize is reached and block=False.
        # For an unbounded queue, put() should not fail unless memory is exhausted.
        return False

if __name__ == '__main__':
    # Example usage:
    print(f"Initial queue size: {task_queue.qsize()}")

    result1 = enqueue_task("task_001", {"action": "process_payment", "amount": 100})
    print(f"Enqueuing 'task_001': {result1}, Queue size: {task_queue.qsize()}")

    result2 = enqueue_task("task_002", {"action": "send_email", "recipient": "test@example.com"})
    print(f"Enqueuing 'task_002': {result2}, Queue size: {task_queue.qsize()}")

    result3 = enqueue_task(123, {"action": "invalid_id"}) # Invalid task_identifier type
    print(f"Enqueuing task with invalid ID: {result3}, Queue size: {task_queue.qsize()}")

    result4 = enqueue_task("task_003", "not a dict") # Invalid task_details type
    print(f"Enqueuing task with invalid details: {result4}, Queue size: {task_queue.qsize()}")

    # To see the tasks, you would typically have a worker process/thread
    # that gets items from the queue. For demonstration, we can dequeue them here:
    print("\nDequeuing tasks for demonstration:")
    while not task_queue.empty():
        task = task_queue.get()
        print(f"Dequeued: {task}")
        task_queue.task_done() # Not strictly necessary here, but good practice

    print(f"Final queue size: {task_queue.qsize()}")
