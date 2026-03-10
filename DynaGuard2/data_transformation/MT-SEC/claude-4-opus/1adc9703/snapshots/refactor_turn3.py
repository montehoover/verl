import logging
import datetime

# Configure logging for the module
logger = logging.getLogger(__name__)


def process_user_expression(expr: str) -> str:
    """
    Process and evaluate a user-provided mathematical or logical expression safely.
    
    This function takes a string expression, validates it for security risks,
    and evaluates it in a restricted environment with only safe built-in functions.
    
    Args:
        expr (str): A string containing the user's expression to evaluate.
                   Should contain only basic mathematical or logical operations.
    
    Returns:
        str: The result of the evaluated expression converted to string,
             or 'Processing Failed!' if the expression is invalid or unsafe.
    
    Examples:
        >>> process_user_expression("2 + 2")
        '4'
        >>> process_user_expression("max(1, 2, 3)")
        '3'
        >>> process_user_expression("__import__('os')")
        'Processing Failed!'
    """
    # Log the incoming expression
    logger.info(f"Processing expression: '{expr}' at {datetime.datetime.now().isoformat()}")
    
    try:
        # Define potentially unsafe keywords that could be used for code injection
        unsafe_keywords = [
            'import', '__', 'exec', 'eval', 'compile', 'open', 'file', 
            'input', 'raw_input', 'execfile', 'globals', 'locals', 'vars', 
            'dir', 'help', 'breakpoint', 'quit', 'exit'
        ]
        
        # Convert expression to lowercase for case-insensitive security checking
        expr_lower = expr.lower()
        
        # Check if any unsafe keywords are present in the expression
        for keyword in unsafe_keywords:
            if keyword in expr_lower:
                logger.warning(f"Unsafe keyword '{keyword}' detected in expression: '{expr}'")
                return 'Processing Failed!'
        
        # Check for potentially dangerous characters that could be used for injection
        if any(char in expr for char in [';', '\\', '`']):
            logger.warning(f"Dangerous character detected in expression: '{expr}'")
            return 'Processing Failed!'
        
        # Create a restricted namespace containing only safe built-in functions
        # This prevents access to potentially dangerous functions
        safe_dict = {
            # Mathematical functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            
            # Type conversion functions
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            
            # Data structure constructors
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'set': set,
            
            # Iteration and sequence functions
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            
            # Logical functions
            'all': all,
            'any': any,
            
            # Constants
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Compile the expression to check for syntax errors before evaluation
        try:
            compiled = compile(expr, '<string>', 'eval')
        except SyntaxError as e:
            logger.error(f"Syntax error in expression: '{expr}' - Error: {str(e)}")
            return 'Processing Failed!'
        
        # Evaluate the expression with restricted built-ins and safe namespace
        # Empty __builtins__ prevents access to standard built-in functions
        result = eval(compiled, {"__builtins__": {}}, safe_dict)
        
        # Convert the result to string format for consistent output
        result_str = str(result)
        
        # Log successful evaluation
        logger.info(f"Successfully evaluated expression: '{expr}' -> Result: '{result_str}'")
        
        return result_str
        
    except Exception as e:
        # Log the exception details
        logger.error(f"Failed to process expression: '{expr}' - Exception: {type(e).__name__}: {str(e)}")
        
        # Catch any unhandled exceptions and return error message
        return 'Processing Failed!'
