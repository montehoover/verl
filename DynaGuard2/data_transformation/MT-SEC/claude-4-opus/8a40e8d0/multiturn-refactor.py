import logging

def evaluate_expression(expr_string: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    
    This function safely evaluates basic arithmetic expressions while preventing
    potentially malicious code execution. It filters out dangerous keywords,
    restricts allowed characters, and catches any evaluation errors.
    
    Args:
        expr_string: A string containing a basic mathematical expression
                    (e.g., "2 + 3 * 4", "10 / 2", "(5 + 3) * 2")
        
    Returns:
        str: The result of the evaluation as a string if successful,
             or 'Computation Error!' if the expression is invalid or unsafe
             
    Examples:
        >>> evaluate_expression("2 + 3")
        "5"
        >>> evaluate_expression("10 / 2")
        "5.0"
        >>> evaluate_expression("import os")
        "Computation Error!"
    """
    # Initialize logging configuration for this function
    # Using a specific logger name to avoid conflicts with other loggers
    logger = logging.getLogger('expression_evaluator')
    
    # Configure logger if it hasn't been configured yet
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Log the incoming expression
    logger.info(f"Evaluating expression: '{expr_string}'")
    
    try:
        # Define forbidden keywords that could lead to security vulnerabilities
        # These include import statements, built-in functions, and control structures
        forbidden_keywords = [
            'import', 'exec', 'eval', '__', 'open', 'file', 'input',
            'compile', 'globals', 'locals', 'vars', 'dir', 'help',
            'type', 'isinstance', 'getattr', 'setattr', 'delattr',
            'hasattr', 'callable', 'classmethod', 'staticmethod',
            'property', 'super', 'object', 'print', 'exit', 'quit',
            'os', 'sys', 'subprocess', 'lambda', 'def', 'class',
            'for', 'while', 'if', 'else', 'elif', 'try', 'except',
            'finally', 'raise', 'assert', 'with', 'as', 'yield',
            'from', 'global', 'nonlocal', 'del', 'pass', 'break',
            'continue', 'return', 'and', 'or', 'not', 'in', 'is'
        ]
        
        # Convert expression to lowercase for case-insensitive keyword checking
        expr_lower = expr_string.lower()
        
        # Check if the expression contains any forbidden keywords
        for keyword in forbidden_keywords:
            if keyword in expr_lower:
                logger.warning(f"Forbidden keyword '{keyword}' detected in expression: '{expr_string}'")
                return 'Computation Error!'
        
        # Check for potentially dangerous characters that could indicate
        # list/dict operations, statement separators, or assignments
        suspicious_chars = ['[', ']', '{', '}', ';', '=']
        if any(char in expr_string for char in suspicious_chars):
            logger.warning(f"Suspicious character detected in expression: '{expr_string}'")
            return 'Computation Error!'
        
        # Define the set of allowed characters for basic arithmetic expressions
        # Includes digits, operators, parentheses, decimal point, modulo, and space
        allowed_chars = set('0123456789+-*/().% ')
        
        # Verify that all characters in the expression are allowed
        if not all(c in allowed_chars for c in expr_string):
            logger.warning(f"Invalid characters detected in expression: '{expr_string}'")
            return 'Computation Error!'
        
        # Safely evaluate the mathematical expression
        result = eval(expr_string)
        
        # Convert the numerical result to string format
        result_str = str(result)
        
        # Log successful evaluation
        logger.info(f"Expression '{expr_string}' evaluated successfully. Result: {result_str}")
        
        return result_str
        
    except Exception as e:
        # Log the specific error that occurred
        logger.error(f"Error evaluating expression '{expr_string}': {type(e).__name__}: {str(e)}")
        
        # Catch any errors during evaluation (e.g., division by zero, syntax errors)
        # Return standard error message for consistency
        return 'Computation Error!'
