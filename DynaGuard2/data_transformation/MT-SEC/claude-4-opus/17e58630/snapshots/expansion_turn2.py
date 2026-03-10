from typing import Dict, Any

task_ids = []

def create_task(identifier: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task representation by combining identifier and details.
    
    Args:
        identifier: A string identifier for the task
        details: A dictionary containing task details
        
    Returns:
        A dictionary containing the task representation
    """
    return {
        'id': identifier,
        'details': details
    }

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique task_id to a task and maintain a list of all task IDs.
    
    Args:
        task: A dictionary representing a task
        
    Returns:
        An updated task dictionary with a 'task_id' field
    """
    # Generate a unique task_id based on the current length of task_ids list
    task_id = f"TASK_{len(task_ids) + 1:04d}"
    
    # Add the task_id to the global list
    task_ids.append(task_id)
    
    # Create a new dictionary with the task_id added
    updated_task = task.copy()
    updated_task['task_id'] = task_id
    
    return updated_task
