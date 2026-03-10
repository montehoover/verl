from typing import Dict, Any
from queue import Queue

tasks = [] # This global list is from the previous version, will be kept for now.
task_queue = Queue()

def register_new_task(task_reference: str, parameters: Dict[str, Any]) -> bool:
    """
    Registers a new task by adding it to the global task queue.

    Args:
        task_reference: The unique ID or reference for the task.
        parameters: A dictionary containing task details.

    Returns:
        True if the task was successfully added to the queue, False otherwise.
    """
    if not isinstance(task_reference, str) or not isinstance(parameters, dict):
        return False
    
    try:
        task_data = {"reference": task_reference, "parameters": parameters}
        task_queue.put(task_data)
        return True
    except Exception:
        # In a real-world scenario, log the exception
        return False

def add_task(task_id: str, task_description: str, task_info: Dict[str, Any]) -> bool:
    """
    Adds a task to the global tasks list. (Kept for context, but new focus is register_new_task)

    Args:
        task_id: The ID of the task.
        task_description: The description of the task.
        task_info: A dictionary containing additional task details like 'priority' and 'deadline'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_id, str) or \
       not isinstance(task_description, str) or \
       not isinstance(task_info, dict):
        return False
    
    # For simplicity, we'll assume task_id should be unique later if needed.
    # For now, just add it.
    task = {"id": task_id, "description": task_description, "info": task_info}
    tasks.append(task)
    return True

if __name__ == '__main__':
    # Example Usage
    print(f"Initial tasks: {tasks}")
    
    result1 = add_task("1", "Buy groceries", {"priority": "High", "deadline": "2025-05-24"})
    print(f"Adding task '1': {result1}, Tasks: {tasks}")
    
    result2 = add_task("2", "Read a book", {"priority": "Medium", "deadline": "2025-05-30"})
    print(f"Adding task '2': {result2}, Tasks: {tasks}")
    
    result3 = add_task(3, "This should fail", {"priority": "Low"}) # type: ignore
    print(f"Adding task with non-string ID: {result3}, Tasks: {tasks}")

    result4 = add_task("4", "Go for a run", {"priority": "High", "deadline": "2025-05-23"})
    print(f"Adding task '4': {result4}, Tasks: {tasks}")

    result5 = add_task("5", "Plan weekend", "not a dict") # type: ignore
    print(f"Adding task with non-dict info: {result5}, Tasks: {tasks}")

    print("\n--- Testing register_new_task ---")
    
    reg_result1 = register_new_task("task_ref_001", {"type": "data_processing", "priority": "High"})
    print(f"Registering task 'task_ref_001': {reg_result1}")

    reg_result2 = register_new_task("task_ref_002", {"type": "email_notification", "recipient": "user@example.com"})
    print(f"Registering task 'task_ref_002': {reg_result2}")

    reg_result3 = register_new_task(123, {"type": "should_fail"}) # type: ignore
    print(f"Registering task with non-string reference: {reg_result3}")

    reg_result4 = register_new_task("task_ref_003", "not a dict") # type: ignore
    print(f"Registering task with non-dict parameters: {reg_result4}")

    print(f"\nTasks in queue (size: {task_queue.qsize()}):")
    while not task_queue.empty():
        print(task_queue.get())
