import re

def is_expression_safe(expression):
    """
    Check if an expression contains only safe arithmetic and logical operations.
    
    Returns True if the expression is safe, False otherwise.
    """
    # Remove whitespace for easier parsing
    expression = expression.replace(" ", "")
    
    # Define allowed characters and patterns
    # Numbers, basic arithmetic operators, parentheses, comparison operators, logical operators
    allowed_pattern = r'^[0-9+\-*/().<>=!&|^~%]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # List of dangerous patterns to check for
    dangerous_patterns = [
        r'__',  # Double underscore (could access special attributes)
        r'import',  # Import statements
        r'exec',  # Execute arbitrary code
        r'eval',  # Evaluate expressions
        r'compile',  # Compile code
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # User input
        r'raw_input',  # User input (Python 2)
        r'\[',  # List/array access
        r'\]',  # List/array access
        r'\{',  # Dictionary/set literals
        r'\}',  # Dictionary/set literals
        r'lambda',  # Lambda functions
        r'def',  # Function definitions
        r'class',  # Class definitions
        r'global',  # Global variable access
        r'locals',  # Local variable access
        r'globals',  # Global variable access
        r'vars',  # Variable access
        r'dir',  # Directory listing
        r'getattr',  # Attribute access
        r'setattr',  # Attribute setting
        r'delattr',  # Attribute deletion
        r'hasattr',  # Attribute checking
        r'callable',  # Check if callable
        r'isinstance',  # Type checking
        r'issubclass',  # Class hierarchy checking
        r'type',  # Type checking
        r'help',  # Help system
        r'print',  # Print function
        r'exit',  # Exit program
        r'quit',  # Quit program
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            return False
    
    # Check for function calls (anything followed by parentheses that isn't a number)
    function_call_pattern = r'[a-zA-Z_]\w*\s*\('
    if re.search(function_call_pattern, expression):
        return False
    
    # Check for attribute access (dots not used for decimals)
    # This pattern looks for dots that aren't between digits
    attribute_pattern = r'(?<![0-9])\.(?![0-9])|(?<=[0-9])\.(?![0-9])|(?<![0-9])\.(?=[0-9])'
    if re.search(attribute_pattern, expression):
        return False
    
    return True
