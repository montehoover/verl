# Global list to store tasks
tasks = []

def add_task(task_name):
    """
    Add a task to the global tasks list.
    
    Args:
        task_name (str): The name of the task to add
        
    Returns:
        bool: True if task is added successfully, False otherwise
    """
    if isinstance(task_name, str) and task_name.strip():
        tasks.append(task_name.strip())
        return True
    return False
