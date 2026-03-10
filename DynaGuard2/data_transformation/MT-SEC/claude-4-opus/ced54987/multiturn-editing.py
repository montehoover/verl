def evaluate_math_expression(expression):
    """
    Evaluates a simple arithmetic expression string and returns the result.
    
    Args:
        expression: A string containing a mathematical expression
        
    Returns:
        The calculated result or 'Error!' if the expression is invalid/unsafe
    """
    try:
        # Remove leading/trailing whitespace
        expression = expression.strip()
        
        # Check for empty expression
        if not expression:
            return "Error!"
        
        # Check if it's a string operation
        if "'" in expression or '"' in expression:
            # For string operations, we need to be more careful
            # Only allow specific characters for string operations
            allowed_string_chars = "0123456789+-*/().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '\"+"
            if not all(c in allowed_string_chars or c.isspace() for c in expression):
                return "Error!"
        else:
            # For numeric operations, remove all whitespace
            expression = expression.replace(" ", "")
            
            # Check for invalid characters in numeric expressions
            allowed_chars = "0123456789+-*/()."
            if not all(c in allowed_chars for c in expression):
                return "Error!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return the result
        return result
        
    except:
        return "Error!"


def process_user_query(user_input):
    """
    Evaluates a user-provided query with basic arithmetic or string operations.
    
    Args:
        user_input: A string containing the query to process
        
    Returns:
        The result of the query execution or an error message
    """
    try:
        # Remove leading/trailing whitespace
        user_input = user_input.strip()
        
        # Check for empty input
        if not user_input:
            return "Error: Empty query"
        
        # Check if it's a string operation
        if "'" in user_input or '"' in user_input:
            # For string operations, only allow safe characters
            allowed_chars = "0123456789+-*/().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '\"+ "
            if not all(c in allowed_chars for c in user_input):
                return "Error: Invalid characters in query"
        else:
            # For numeric operations, check for valid characters
            allowed_chars = "0123456789+-*/(). "
            if not all(c in allowed_chars for c in user_input):
                return "Error: Invalid characters in query"
        
        # Evaluate the query
        result = eval(user_input)
        
        # Return the result
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"
