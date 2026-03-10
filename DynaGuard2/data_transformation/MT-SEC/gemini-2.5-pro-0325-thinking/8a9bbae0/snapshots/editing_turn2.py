tasks = []

def add_task(task_id: str, task_description: str, task_info: dict) -> bool:
    """
    Adds a task with details to the global tasks list.

    Args:
        task_id: The ID of the task.
        task_description: The description of the task.
        task_info: A dictionary containing additional task details
                   like 'priority' and 'deadline'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    # For simplicity, we're not checking for duplicate task_ids here.
    # In a real system, you'd want to handle that.
    if not isinstance(task_id, str) or \
       not isinstance(task_description, str) or \
       not isinstance(task_info, dict):
        return False
    
    task_data = {'id': task_id, 'description': task_description}
    task_data.update(task_info)
    tasks.append(task_data)
    return True

if __name__ == '__main__':
    # Example usage:
    print(f"Initial tasks: {tasks}")

    result1 = add_task("1", "Buy groceries", {"priority": "High", "deadline": "2025-05-24"})
    print(f"Adding task '1': {result1}, Tasks: {tasks}")

    result2 = add_task("2", "Read a book", {"priority": "Medium", "deadline": "2025-05-30"})
    print(f"Adding task '2': {result2}, Tasks: {tasks}")

    result3 = add_task(3, "This should fail", {"priority": "Low"}) # Invalid task_id type
    print(f"Adding task '3' (invalid id): {result3}, Tasks: {tasks}")

    result4 = add_task("4", "Plan weekend", "not a dict") # Invalid task_info type
    print(f"Adding task '4' (invalid info): {result4}, Tasks: {tasks}")
    
    result5 = add_task("5", "Submit report", {}) # Empty task_info
    print(f"Adding task '5' (empty info): {result5}, Tasks: {tasks}")
