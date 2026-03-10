def execute_operation(operation):
    """Execute a simple arithmetic operation given as a string.
    
    Args:
        operation: A string containing a simple arithmetic operation (e.g., '5 + 3')
        
    Returns:
        The result of the arithmetic operation
    """
    # Create a local namespace for execution
    local_namespace = {}
    
    # Split the operation into individual statements
    statements = operation.split(';')
    
    # Execute each statement
    result = None
    for statement in statements:
        statement = statement.strip()
        if statement:
            result = eval(statement, {"__builtins__": {}}, local_namespace)
            # If it's an assignment, update the namespace
            if '=' in statement and not any(op in statement for op in ['==', '!=', '<=', '>=']):
                exec(statement, {"__builtins__": {}}, local_namespace)
    
    return result
