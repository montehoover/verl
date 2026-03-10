import ast
import operator

def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a mathematical expression string without using eval() or exec().
    
    Args:
        math_expression: A string representing a mathematical expression
        
    Returns:
        str: The result of the evaluation or 'Unsafe Code Detected!' if unsafe
    """
    
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Define allowed functions
    allowed_functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
    }
    
    def safe_eval(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Handle numbers and constants
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations like -x
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError("Unsafe operation detected")
            return op_func(safe_eval(node.operand))
        elif isinstance(node, ast.BinOp):
            # Handle binary operations like x + y
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError("Unsafe operation detected")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return op_func(left, right)
        elif isinstance(node, ast.Call):
            # Handle function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in allowed_functions:
                    raise ValueError("Unsafe function detected")
                args = [safe_eval(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            else:
                raise ValueError("Unsafe function call detected")
        elif isinstance(node, ast.Name):
            # Variables are not allowed
            raise ValueError("Variables not allowed")
        else:
            raise ValueError("Unsafe construct detected")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(math_expression, mode='eval')
        
        # Evaluate the expression safely
        result = safe_eval(tree.body)
        
        # Return the result as a string
        return str(result)
        
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, AttributeError):
        return "Unsafe Code Detected!"
    except Exception:
        return "Unsafe Code Detected!"
