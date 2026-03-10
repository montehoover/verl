from typing import Dict, Any


existing_task_ids = []


def create_task(task_reference: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a task object that combines task reference and parameters.
    
    Args:
        task_reference: A string identifier for the task
        parameters: A dictionary containing task parameters
        
    Returns:
        A dictionary containing the task reference and parameters
    """
    return {
        "task_reference": task_reference,
        "parameters": parameters
    }


def assign_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task_id to a task dictionary.
    
    Args:
        task: A dictionary representing a task
        
    Returns:
        The task dictionary with a unique task_id added
    """
    # Generate a unique task_id
    task_id = 1
    while task_id in existing_task_ids:
        task_id += 1
    
    # Add the task_id to the global list
    existing_task_ids.append(task_id)
    
    # Add the task_id to the task dictionary
    task["task_id"] = task_id
    
    return task
