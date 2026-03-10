import math

def calculate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for invalid characters
        valid_chars = "0123456789+-*/()."
        for char in expression:
            if char not in valid_chars:
                return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Handle case where result is zero or negative (log undefined)
        if result <= 0:
            return "Invalid Expression!"
        
        # Return the log of the result
        return math.log(result)
    except:
        return "Invalid Expression!"

def process_user_expression(expr):
    try:
        # Remove whitespace
        expr = expr.replace(" ", "")
        
        # Check for invalid characters - only allow basic math operations
        valid_chars = "0123456789+-*/()."
        for char in expr:
            if char not in valid_chars:
                return "Processing Failed!"
        
        # Additional security checks
        # Block common dangerous patterns
        dangerous_patterns = ["__", "import", "exec", "eval", "compile", "open", "file", "input", "raw_input"]
        expr_lower = expr.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                return "Processing Failed!"
        
        # Evaluate the expression
        result = eval(expr)
        return result
    except:
        return "Processing Failed!"
