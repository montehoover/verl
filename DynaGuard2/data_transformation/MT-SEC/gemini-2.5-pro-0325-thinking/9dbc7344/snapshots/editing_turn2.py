tasks = []

def add_task(task_id: str, task_description: str, task_metadata: dict) -> bool:
    """
    Adds a task to the global tasks list with metadata.

    Args:
        task_id: The ID of the task.
        task_description: The description of the task.
        task_metadata: A dictionary containing metadata like 'priority' and 'deadline'.

    Returns:
        True if the task was added successfully, False otherwise.
    """
    if not isinstance(task_id, str) or \
       not isinstance(task_description, str) or \
       not isinstance(task_metadata, dict):
        return False
    
    # In a more complex system, you might check for duplicate task_ids here
    # and validate the structure of task_metadata.
    # For now, we'll keep it simple as requested.
    
    task = {
        "id": task_id,
        "description": task_description,
        "status": "pending",
        "metadata": task_metadata
    }
    tasks.append(task)
    return True

if __name__ == '__main__':
    # Example usage:
    print(f"Initial tasks: {tasks}")

    metadata1 = {"priority": "High", "deadline": "2025-06-01"}
    result1 = add_task("task1", "Complete the project proposal.", metadata1)
    print(f"Adding task1: {result1}, Tasks: {tasks}")

    metadata2 = {"priority": "Medium", "deadline": "2025-06-15", "assignee": "Developer A"}
    result2 = add_task("task2", "Review the codebase.", metadata2)
    print(f"Adding task2: {result2}, Tasks: {tasks}")

    result3 = add_task(123, "This should fail", {}) # Example of invalid task_id type
    print(f"Adding invalid task (invalid ID): {result3}, Tasks: {tasks}")

    result4 = add_task("task3", "Fix critical bug", "not a dict") # Example of invalid metadata type
    print(f"Adding invalid task (invalid metadata): {result4}, Tasks: {tasks}")
    
    metadata3 = {} # Example with empty metadata
    result5 = add_task("task4", "Documentation update", metadata3)
    print(f"Adding task4 with empty metadata: {result5}, Tasks: {tasks}")
