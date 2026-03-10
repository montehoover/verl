from typing import Dict, Any


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
