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
    
    # In a more complex system, you might check for duplicate task_ids here.
    # For now, we'll keep it simple as requested.
    
    task = {"id": task_id, "description": task_description, "status": "pending"}
    tasks.append(task)
    return True

if __name__ == '__main__':
    # Example usage:
    print(f"Initial tasks: {tasks}")

    result1 = add_task("task1", "Complete the project proposal.")
    print(f"Adding task1: {result1}, Tasks: {tasks}")

    result2 = add_task("task2", "Review the codebase.")
    print(f"Adding task2: {result2}, Tasks: {tasks}")

    result3 = add_task(123, "This should fail") # Example of invalid input
    print(f"Adding invalid task: {result3}, Tasks: {tasks}")
