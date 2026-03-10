import re
import ast


def parse_and_calculate(expression: str):
    """Parse and calculate a mathematical expression safely.
    
    This function takes a string containing a mathematical expression,
    validates it for safety, and returns the calculated result.
    
    Args:
        expression: A string containing the arithmetic expression to evaluate.
                   Supports basic operators (+, -, *, /), parentheses, and decimals.
    
    Returns:
        The numeric result of the calculated expression.
    
    Raises:
        ValueError: If the expression is empty, contains unsupported characters,
                   potentially malicious commands, or cannot be computed.
    """
    # Input validation
    cleaned_expression = expression.strip()
    
    if not cleaned_expression:
        raise ValueError("Empty expression")
    
    # Regex pattern explanation:
    # ^ - start of string
    # [0-9+\-*/().\s]+ - one or more of: digits, operators (+,-,*,/), parentheses, decimal points, whitespace
    # $ - end of string
    allowed_characters_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_characters_pattern, cleaned_expression):
        raise ValueError("Expression contains unsupported characters")
    
    # Security checks for potentially malicious patterns
    suspicious_keywords = ['__', 'import', 'eval', 'exec']
    for keyword in suspicious_keywords:
        if keyword in cleaned_expression:
            raise ValueError("Expression contains potentially malicious commands")
    
    try:
        # Parse and validate AST structure
        syntax_tree = ast.parse(cleaned_expression, mode='eval')
        
        # Ensure only arithmetic operations are present in the AST
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
            elif isinstance(node, ast.Name):
                raise ValueError("Variable names are not allowed")
        
        # Compile and execute the validated expression
        compiled_code = compile(syntax_tree, '<string>', 'eval')
        result = eval(compiled_code)
        
        # Validate the result for mathematical errors
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                raise ValueError("Cannot compute: division by zero or invalid operation")
            if result != result:  # Check for NaN
                raise ValueError("Cannot compute: division by zero or invalid operation")
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Cannot compute: division by zero")
    except Exception:
        raise ValueError("Cannot compute the expression")
