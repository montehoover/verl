import math

def evaluate_expression(expression, variables=None):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # If variables provided, substitute them
        if variables:
            for var_name, var_value in variables.items():
                expression = expression.replace(var_name, str(var_value))
        
        # Validate expression contains only allowed characters
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Evaluate the expression
        result = eval(expression)
        
        # Take the log of the result
        if result <= 0:
            raise ValueError("Cannot take log of non-positive number")
        log_result = math.log(result)
        
        # Return result as string
        return str(log_result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
