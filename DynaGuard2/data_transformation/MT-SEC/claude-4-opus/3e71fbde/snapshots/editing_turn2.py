def execute_operation(operation):
    """
    Execute a simple arithmetic operation or complex Python expression from a string.
    
    Args:
        operation (str): A string representing a Python expression (e.g., '2 + 3' or 'a = 5; b = a * 2')
    
    Returns:
        int, float, or any: The result of the expression execution
    """
    # Create a local namespace for executing the expression
    local_namespace = {}
    
    # Execute the expression
    exec(operation, {}, local_namespace)
    
    # If it's a single expression without assignment, evaluate and return it
    if '=' not in operation and ';' not in operation:
        return eval(operation, {}, local_namespace)
    
    # For multi-statement expressions, try to find the last expression's value
    statements = [s.strip() for s in operation.split(';') if s.strip()]
    if statements:
        last_statement = statements[-1]
        if '=' not in last_statement:
            return eval(last_statement, {}, local_namespace)
        else:
            # If the last statement is an assignment, return the assigned value
            var_name = last_statement.split('=')[0].strip()
            return local_namespace.get(var_name)
    
    return None
