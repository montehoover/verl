from typing import Dict, Any
from queue import Queue

task_ids = []
task_queue = Queue()

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

def enqueue_task(identifier: str, details: Dict[str, Any]) -> bool:
    """
    Integrate a task into the global queue for concurrent processing.
    
    Args:
        identifier: A string identifier for the task
        details: A dictionary containing task details
        
    Returns:
        A boolean indicating success of the enqueue operation
    """
    try:
        # Create the task
        task = create_task(identifier, details)
        
        # Generate a unique task_id for the task
        task_with_id = generate_task_id(task)
        
        # Add the task to the queue
        task_queue.put(task_with_id)
        
        return True
    except Exception:
        return False
