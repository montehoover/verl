import ast
import math

def safely_execute_expression(expression_code: str) -> str:
    """
    Safely execute a mathematical expression written in Python.
    
    Args:
        expression_code: A string representing a user-input Python expression for calculation
        
    Returns:
        Either the result of the secure calculation or an error message flagging unsafe code
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_code, mode='eval')
        
        # Define allowed node types for safe mathematical expressions
        allowed_nodes = (
            ast.Expression,
            ast.Num,  # Numbers (Python 3.7 and below)
            ast.Constant,  # Constants (Python 3.8+)
            ast.BinOp,  # Binary operations
            ast.UnaryOp,  # Unary operations
            ast.Compare,  # Comparisons
            ast.BoolOp,  # Boolean operations
            ast.Name,  # Variable names (we'll restrict these)
            ast.Load,  # Load context
            # Allowed operators
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd,  # Unary operators
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,  # Comparison operators
            ast.And, ast.Or, ast.Not,  # Boolean operators
        )
        
        # Check all nodes in the AST
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "Unsafe Code Detected!"
            
            # If it's a Name node, only allow specific safe names
            if isinstance(node, ast.Name):
                allowed_names = ['math', 'pi', 'e', 'True', 'False', 'None']
                if node.id not in allowed_names:
                    return "Unsafe Code Detected!"
        
        # Create a safe namespace with only math functions
        safe_namespace = {
            'math': math,
            'pi': math.pi,
            'e': math.e,
            'True': True,
            'False': False,
            'None': None,
            '__builtins__': {}  # Empty builtins to prevent access to dangerous functions
        }
        
        # Compile and evaluate the expression
        compiled_code = compile(tree, '<string>', 'eval')
        result = eval(compiled_code, safe_namespace)
        
        return str(result)
        
    except Exception:
        return "Unsafe Code Detected!"
