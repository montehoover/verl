from typing import Dict, Any

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
