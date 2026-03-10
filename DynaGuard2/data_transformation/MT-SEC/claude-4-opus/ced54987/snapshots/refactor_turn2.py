def process_user_query(user_input: str) -> str:
    """
    Process a user's query expression and evaluate it safely.
    
    This function takes a string input containing a Python expression,
    evaluates it, and returns the result as a string. It includes
    security checks to prevent execution of potentially dangerous code.
    
    Args:
        user_input (str): The user-provided string to be interpreted 
                         and computed as a Python expression.
    
    Returns:
        str: The string representation of the computed result, or
             'Error!' if the input is deemed untrustworthy or any
             exception occurs during evaluation.
    
    Raises:
        No exceptions are raised; errors return 'Error!' string.
    """
    try:
        # Define keywords that could pose security risks
        dangerous_keywords = [
            '__import__', 'exec', 'eval', 'open', 'file', 
            'input', 'compile', 'globals', 'locals', '__', 'import'
        ]
        
        # Check if input contains any dangerous operations
        for keyword in dangerous_keywords:
            if keyword in user_input:
                return 'Error!'
        
        # Evaluate the expression if it passes security checks
        result = eval(user_input)
        
        # Convert result to string and return
        return str(result)
        
    except:
        # Return error message for any exception during evaluation
        return 'Error!'
