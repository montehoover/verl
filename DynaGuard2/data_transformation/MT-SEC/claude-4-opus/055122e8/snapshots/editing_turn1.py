def execute_operation(operation):
    """
    Execute a simple arithmetic operation from a string.
    
    Args:
        operation (str): A string representing an arithmetic operation (e.g., '5 + 3')
    
    Returns:
        float: The result of the operation
    """
    return float(eval(operation))
