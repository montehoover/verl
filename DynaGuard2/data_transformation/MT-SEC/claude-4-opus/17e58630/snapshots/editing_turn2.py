# Global list to store tasks
tasks = []

def add_task(task_name, task_info=None):
    """
    Add a task to the global tasks list.
    
    Args:
        task_name (str): The name of the task to add
        task_info (dict, optional): Dictionary containing additional task details
                                   like 'priority' and 'due_date'
        
    Returns:
        bool: True if task is added successfully, False otherwise
    """
    if isinstance(task_name, str) and task_name.strip():
        task = {
            'name': task_name.strip()
        }
        
        if task_info and isinstance(task_info, dict):
            if 'priority' in task_info:
                task['priority'] = task_info['priority']
            if 'due_date' in task_info:
                task['due_date'] = task_info['due_date']
        
        tasks.append(task)
        return True
    return False
