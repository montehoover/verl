# Global list to store tasks
tasks = []

def add_task(task_id, task_description):
    """
    Add a task to the global tasks list.
    
    Args:
        task_id (str): The unique identifier for the task
        task_description (str): The description of the task
    
    Returns:
        bool: True if task was added successfully, False otherwise
    """
    try:
        # Create task dictionary
        task = {
            'id': task_id,
            'description': task_description
        }
        
        # Check if task_id already exists
        for existing_task in tasks:
            if existing_task['id'] == task_id:
                return False
        
        # Add task to list
        tasks.append(task)
        return True
    except:
        return False
