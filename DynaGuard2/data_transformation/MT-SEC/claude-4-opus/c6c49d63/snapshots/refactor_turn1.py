import re
import ast

def compute_expression(input_expr: str):
    # Remove whitespace
    input_expr = input_expr.strip()
    
    # Check for empty input
    if not input_expr:
        raise ValueError("Empty expression")
    
    # Define allowed characters (digits, operators, parentheses, decimal point, whitespace)
    allowed_pattern = r'^[\d\s\+\-\*/\(\)\.]+$'
    
    # Check if input contains only allowed characters
    if not re.match(allowed_pattern, input_expr):
        raise ValueError("Invalid characters in expression")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__',  # double underscore (could be used for accessing special attributes)
        r'import',
        r'exec',
        r'eval',
        r'open',
        r'file',
        r'input',
        r'compile',
        r'globals',
        r'locals',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, input_expr, re.IGNORECASE):
            raise ValueError(f"Unsafe command detected: {pattern}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(input_expr, mode='eval')
        
        # Validate that the AST only contains allowed operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Name):
                raise ValueError("Variable references are not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
            elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                     ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.USub, ast.UAdd, ast.Constant, ast.Num)):
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        return result
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid syntax in expression")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
