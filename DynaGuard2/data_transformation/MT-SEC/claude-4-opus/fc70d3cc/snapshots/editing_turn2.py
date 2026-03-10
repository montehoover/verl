def execute_operation(operation_str):
    """
    Execute a simple arithmetic operation from a string.
    
    Args:
        operation_str: A string representing an arithmetic operation (e.g., '2 + 3')
    
    Returns:
        The result of the operation
    """
    # Create a local namespace for the execution
    local_vars = {}
    
    # Execute the expression(s)
    exec(operation_str, {"__builtins__": {}}, local_vars)
    
    # If it's a simple expression without assignment, evaluate it
    if '=' not in operation_str or ';' in operation_str:
        # For multi-statement expressions, get the last expression
        statements = operation_str.split(';')
        last_statement = statements[-1].strip()
        
        # If the last statement is not an assignment, evaluate it
        if last_statement and '=' not in last_statement:
            return eval(last_statement, {"__builtins__": {}}, local_vars)
    
    # For single expressions, try to evaluate directly
    try:
        return eval(operation_str, {"__builtins__": {}}, local_vars)
    except:
        # If evaluation fails, return None for assignment-only statements
        return None
