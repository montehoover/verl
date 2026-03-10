def execute_simple_operation(operation):
    """
    Execute a simple arithmetic operation given as a string.
    
    Args:
        operation (str): A string representing a simple arithmetic operation (e.g., '2 + 3')
    
    Returns:
        int or float: The result of the operation
    """
    return eval(operation)
