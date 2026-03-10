from queue import Queue

task_queue = Queue()

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Inserts a task into the global task queue.

    Args:
        task_identifier: The unique ID of the task.
        task_content: A dictionary containing the task parameters.

    Returns:
        True if the task was inserted successfully, False otherwise.
    """
    if not isinstance(task_identifier, str) or not isinstance(task_content, dict):
        return False
    
    try:
        # In a real system, you might want to ensure task_identifier is unique
        # or add more validation for task_content.
        # For now, we directly put a tuple (identifier, content) into the queue.
        task_queue.put((task_identifier, task_content))
        return True
    except Exception:
        # Catch any potential exceptions during queue insertion, though put() is generally safe.
        return False

if __name__ == '__main__':
    # Example usage:
    print(f"Initial queue size: {task_queue.qsize()}")

    content1 = {"description": "Complete the project proposal.", "priority": "High", "deadline": "2025-06-01"}
    result1 = insert_task_to_queue("task1", content1)
    print(f"Inserting task1: {result1}, Queue size: {task_queue.qsize()}")

    content2 = {"description": "Review the codebase.", "priority": "Medium", "deadline": "2025-06-15", "assignee": "Developer A"}
    result2 = insert_task_to_queue("task2", content2)
    print(f"Inserting task2: {result2}, Queue size: {task_queue.qsize()}")

    result3 = insert_task_to_queue(123, {})  # Example of invalid task_identifier type
    print(f"Inserting invalid task (invalid ID): {result3}, Queue size: {task_queue.qsize()}")

    result4 = insert_task_to_queue("task3", "not a dict")  # Example of invalid task_content type
    print(f"Inserting invalid task (invalid content): {result4}, Queue size: {task_queue.qsize()}")

    # To see the tasks, you would typically have a worker process getting items from the queue.
    # For demonstration, we can show the items (not recommended for direct queue inspection in prod):
    print("Tasks in queue (for demonstration):")
    # Be careful with iterating directly over queue.queue as it's not thread-safe for modification
    # This is just for a simple, single-threaded demonstration.
    temp_list = list(task_queue.queue)
    for item in temp_list:
        print(item)
    
    # Example of retrieving a task (how a worker might do it)
    if not task_queue.empty():
        retrieved_task_id, retrieved_task_content = task_queue.get()
        print(f"\nRetrieved task: ID={retrieved_task_id}, Content={retrieved_task_content}")
        print(f"Queue size after retrieval: {task_queue.qsize()}")
    else:
        print("\nQueue is empty, no task to retrieve.")
