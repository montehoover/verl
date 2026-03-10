import re
import ast

def compute_expression(input_expr: str):
    """
    Compute the result of a mathematical expression provided as a string.
    
    Args:
        input_expr: str, a string representing the arithmetic expression
        
    Returns:
        The computed result of the expression
        
    Raises:
        ValueError: if unsupported characters, unsafe commands, or invalid operations are detected
    """
    # Remove whitespace for easier validation
    cleaned_expr = input_expr.strip()
    
    # Check for empty input
    if not cleaned_expr:
        raise ValueError("Empty expression provided")
    
    # Define allowed characters (digits, operators, parentheses, decimal point, and whitespace)
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, cleaned_expr):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for potentially dangerous patterns
    dangerous_patterns = [
        r'__',  # double underscore (could be used for attribute access)
        r'import',  # import statements
        r'eval',  # eval function
        r'exec',  # exec function
        r'open',  # file operations
        r'file',  # file operations
        r'input',  # input function
        r'raw_input',  # raw_input function (Python 2)
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, cleaned_expr, re.IGNORECASE):
            raise ValueError(f"Expression contains unsafe command: {pattern}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(cleaned_expr, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        }
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid mathematical expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
