# Global list to store tasks
tasks = []

def add_task(task_name, task_info):
    """Add a task to the global tasks list.
    
    Args:
        task_name (str): The name of the task
        task_info (str): Additional information about the task
        
    Returns:
        bool: True if task added successfully, False otherwise
    """
    try:
        task = {
            'name': task_name,
            'info': task_info
        }
        tasks.append(task)
        return True
    except:
        return False
