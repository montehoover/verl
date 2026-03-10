tasks = []

def add_task(task_id: str, task_description: str) -> bool:
    """
    Adds a task to the global tasks list.

    Args:
        task_id: The ID of the task.
        task_description: The description of the task.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    # For simplicity, we're not checking for duplicate task_ids here.
    # In a real system, you'd want to handle that.
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False
    
    tasks.append({'id': task_id, 'description': task_description})
    return True

if __name__ == '__main__':
    # Example usage:
    print(f"Initial tasks: {tasks}")

    result1 = add_task("1", "Buy groceries")
    print(f"Adding task '1': {result1}, Tasks: {tasks}")

    result2 = add_task("2", "Read a book")
    print(f"Adding task '2': {result2}, Tasks: {tasks}")

    result3 = add_task(3, "This should fail") # Invalid task_id type
    print(f"Adding task '3' (invalid): {result3}, Tasks: {tasks}")
