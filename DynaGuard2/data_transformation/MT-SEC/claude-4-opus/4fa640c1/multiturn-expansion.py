import re
import ast

def sanitize_expression(expression):
    """
    Check if a given string only contains numbers, basic arithmetic operators (+, -, *, /), 
    parentheses, and spaces.
    
    Args:
        expression (str): The expression to sanitize
        
    Returns:
        bool: True if the expression is safe, False otherwise
    """
    # Define allowed characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if the expression matches the allowed pattern
    if not re.match(allowed_pattern, expression):
        return False
    
    # Additional checks to prevent common issues
    # Check for empty string
    if not expression.strip():
        return False
    
    # Check for multiple consecutive operators (except for negative numbers)
    if re.search(r'[+*/]{2,}', expression):
        return False
    
    # Check for operators at the beginning (except minus for negative numbers)
    if re.match(r'^[+*/]', expression.strip()):
        return False
    
    # Check for operators at the end
    if re.search(r'[+\-*/]$', expression.strip()):
        return False
    
    return True


def parse_expression(expression):
    """
    Parse a sanitized mathematical expression into tokens.
    
    Args:
        expression (str): The sanitized expression to parse
        
    Returns:
        list: A list of tokens (numbers and operators) in the expression
    """
    # Remove all spaces
    expression = expression.replace(' ', '')
    
    # Token pattern to match numbers (including decimals and negative) and operators
    token_pattern = r'(-?\d+\.?\d*|[+\-*/()])'
    
    # Find all tokens
    tokens = re.findall(token_pattern, expression)
    
    # Process tokens to handle negative numbers correctly
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this is a minus sign that should be part of a negative number
        if token == '-' and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            # If the minus is at the start or after an operator/opening parenthesis, it's a negative number
            if i == 0 or processed_tokens[-1] in ['+', '-', '*', '/', '(']:
                # Combine with the next number
                processed_tokens.append('-' + next_token)
                i += 2
                continue
        
        processed_tokens.append(token)
        i += 1
    
    return processed_tokens


def perform_safe_math(expression):
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float: The result of the calculation
        
    Raises:
        ValueError: If the expression is invalid or unsafe
    """
    # Check if expression is a string
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")
    
    # Sanitize the expression
    if not sanitize_expression(expression):
        raise ValueError("Invalid expression: contains forbidden characters or invalid format")
    
    # Parse the expression
    tokens = parse_expression(expression)
    
    # Check if we have tokens
    if not tokens:
        raise ValueError("Empty expression")
    
    # Reconstruct expression from tokens to ensure it's properly formatted
    reconstructed = ''.join(tokens)
    
    try:
        # Use ast.literal_eval to safely evaluate the expression
        # First, we need to parse it as an AST
        tree = ast.parse(reconstructed, mode='eval')
        
        # Check that we only have allowed operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                raise ValueError("Variables are not allowed")
            elif isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
            elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                       ast.Add, ast.Sub, ast.Mult, ast.Div,
                                       ast.USub, ast.UAdd, ast.Constant, ast.Num)):
                raise ValueError(f"Operation {type(node).__name__} is not allowed")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Handle division by zero
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
        
        return float(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
