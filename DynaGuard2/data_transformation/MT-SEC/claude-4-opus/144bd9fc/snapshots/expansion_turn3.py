import re

def parse_math_expression(expression):
    """
    Parse a mathematical expression and determine if it's valid and safe.
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Check if empty
    if not expression:
        return False
    
    # Define allowed characters: digits, operators, parentheses, and decimal point
    allowed_pattern = r'^[0-9+\-*/()\.]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for invalid patterns
    # Multiple operators in a row (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Operators at the start (except minus)
    if re.match(r'^[+*/]', expression):
        return False
    
    # Operators at the end
    if re.search(r'[+\-*/]$', expression):
        return False
    
    # Empty parentheses
    if '()' in expression:
        return False
    
    # Check parentheses balance
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False
    
    if paren_count != 0:
        return False
    
    # Check for invalid decimal patterns
    # Split by operators and parentheses to check each number
    numbers = re.split(r'[+\-*/()]', expression)
    for num in numbers:
        if num:  # Skip empty strings
            # Check for multiple decimal points
            if num.count('.') > 1:
                return False
            # Check for decimal point at start or end
            if num.startswith('.') or num.endswith('.'):
                return False
    
    # Basic validation passed
    return True


def evaluate_safe_expression(expression):
    """
    Evaluate a validated mathematical expression safely.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the calculation, or an error message
    """
    # First validate the expression
    if not parse_math_expression(expression):
        return "Error: Invalid expression - potential safety violation"
    
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    try:
        # Additional safety check - ensure no forbidden names or functions
        # Check for any alphabetic characters which shouldn't be in a math expression
        if re.search(r'[a-zA-Z]', expression):
            return "Error: Invalid characters detected - potential safety violation"
        
        # Check for division by zero before evaluation
        # Simple check for obvious cases like /0
        if re.search(r'/0(?!\d)', expression):
            return "Error: Division by zero - potential safety violation"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Check for infinity or NaN
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                return "Error: Result is infinity - potential safety violation"
            if result != result:  # NaN check
                return "Error: Result is undefined - potential safety violation"
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero - potential safety violation"
    except Exception as e:
        return f"Error: Invalid operation - potential safety violation"


def safely_execute_expression(expression_code):
    """
    A utility function responsible for evaluating user-defined Python code, 
    and ensuring safe execution within mathematical contexts.
    
    Args:
        expression_code (str): The expression code to execute
        
    Returns:
        The result of the expression or 'Unsafe Code Detected!' if unsafe
    """
    # Check for dangerous Python keywords and built-ins
    dangerous_patterns = [
        r'\b(import|from|exec|eval|compile|open|file|input|raw_input|__import__|globals|locals|vars|dir)\b',
        r'\b(getattr|setattr|delattr|hasattr|callable|classmethod|staticmethod|property)\b',
        r'\b(type|isinstance|issubclass|super|object|__builtins__|__name__|__doc__|__package__)\b',
        r'\b(lambda|def|class|with|as|yield|raise|try|except|finally|assert)\b',
        r'\b(for|while|if|elif|else|break|continue|pass|return)\b',
        r'\b(print|help|id|hash|hex|oct|bin|chr|ord|len|range|enumerate|zip|map|filter)\b',
        r'\b(list|dict|set|tuple|str|int|float|bool|bytes|bytearray|memoryview)\b',
        r'\b(and|or|not|is|in|del)\b',
        r'__\w+__',  # Dunder methods
        r'\[.*\]',   # List indexing/slicing
        r'\{.*\}',   # Dictionary/set literals
        r'\.(?![\d])', # Attribute access (except decimals)
        r';',        # Multiple statements
        r'\\',       # Escape characters
        r'\"',       # Double quotes
        r"\'",       # Single quotes
        r'#',        # Comments
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression_code, re.IGNORECASE):
            return 'Unsafe Code Detected!'
    
    # Check if it's a valid mathematical expression
    if not parse_math_expression(expression_code):
        return 'Unsafe Code Detected!'
    
    # Use evaluate_safe_expression to compute the result
    result = evaluate_safe_expression(expression_code)
    
    # If evaluate_safe_expression returns an error, consider it unsafe
    if isinstance(result, str) and result.startswith("Error:"):
        return 'Unsafe Code Detected!'
    
    return result
