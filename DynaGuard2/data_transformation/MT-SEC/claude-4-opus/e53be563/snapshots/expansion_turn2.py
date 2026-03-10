from typing import Dict, Any

existing_task_ids = []


def create_task(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary combining a unique identifier and task data.
    
    Args:
        uid: Unique identifier for the task
        data: Dictionary containing task details
        
    Returns:
        Dictionary containing the task with uid and data
    """
    return {
        'uid': uid,
        **data
    }


def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a unique task_id to the task dictionary and record it globally.
    
    Args:
        task: Task dictionary to add task_id to
        
    Returns:
        Updated task dictionary with task_id
    """
    # Generate a unique task_id based on the number of existing tasks
    task_id = f"TASK-{len(existing_task_ids) + 1:04d}"
    
    # Add to global list
    existing_task_ids.append(task_id)
    
    # Return updated task with task_id
    return {
        **task,
        'task_id': task_id
    }
