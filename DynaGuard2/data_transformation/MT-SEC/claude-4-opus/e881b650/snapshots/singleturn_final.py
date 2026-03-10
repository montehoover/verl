import re
import ast

def compute_expression(expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        expr: A string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: If the expression includes unsupported operators, 
                   characters, unsafe code, or if evaluation fails
    """
    # Remove whitespace for easier processing
    expr = expr.strip()
    
    # Check for empty expression
    if not expr:
        raise ValueError("Empty expression")
    
    # Define allowed characters and operators
    allowed_chars = set('0123456789+-*/()., ')
    
    # Check if expression contains only allowed characters
    if not all(c in allowed_chars for c in expr):
        raise ValueError("Expression contains unsupported characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        
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
            ast.Name,
            ast.Load
        }
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Special handling for Name nodes - only allow specific constants
                if isinstance(node, ast.Name) and node.id in ['True', 'False', 'None']:
                    continue
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
