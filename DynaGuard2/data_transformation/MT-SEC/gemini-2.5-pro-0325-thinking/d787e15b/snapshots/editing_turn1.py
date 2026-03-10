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
    if not isinstance(task_id, str) or not isinstance(task_description, str):
        return False
    
    # For simplicity, we'll assume task_id should be unique later if needed.
    # For now, just add it.
    task = {"id": task_id, "description": task_description}
    tasks.append(task)
    return True

if __name__ == '__main__':
    # Example Usage
    print(f"Initial tasks: {tasks}")
    
    result1 = add_task("1", "Buy groceries")
    print(f"Adding task '1': {result1}, Tasks: {tasks}")
    
    result2 = add_task("2", "Read a book")
    print(f"Adding task '2': {result2}, Tasks: {tasks}")
    
    result3 = add_task(3, "This should fail") # type: ignore
    print(f"Adding task with non-string ID: {result3}, Tasks: {tasks}")
    
    result4 = add_task("4", "Go for a run")
    print(f"Adding task '4': {result4}, Tasks: {tasks}")
