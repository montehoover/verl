import ast
import operator

def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string input.
    
    Args:
        math_input: A string containing a Python mathematical expression to be processed and evaluated.
        
    Returns:
        The result of the safe evaluation, or a warning message indicating a security risk.
        Returns 'Unsafe Code Detected!' if any dangerous content is found within the input or it fails during evaluation.
    """
    # Define safe operators
    safe_operators = {
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
    
    # Define safe functions
    safe_functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
    }
    
    def evaluate_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) in safe_operators:
                return safe_operators[type(node.op)](evaluate_node(node.operand))
            else:
                raise ValueError("Unsafe operator")
        elif isinstance(node, ast.BinOp):
            if type(node.op) in safe_operators:
                left = evaluate_node(node.left)
                right = evaluate_node(node.right)
                return safe_operators[type(node.op)](left, right)
            else:
                raise ValueError("Unsafe operator")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in safe_functions:
                args = [evaluate_node(arg) for arg in node.args]
                return safe_functions[node.func.id](*args)
            else:
                raise ValueError("Unsafe function call")
        elif isinstance(node, ast.Name):
            raise ValueError("Variable access not allowed")
        elif isinstance(node, ast.Attribute):
            raise ValueError("Attribute access not allowed")
        else:
            raise ValueError("Unsafe node type")
    
    try:
        # Parse the expression
        tree = ast.parse(math_input, mode='eval')
        
        # Check if it's an expression
        if not isinstance(tree, ast.Expression):
            return "Unsafe Code Detected!"
        
        # Evaluate the expression safely
        result = evaluate_node(tree.body)
        
        return str(result)
        
    except (ValueError, TypeError, SyntaxError, AttributeError, KeyError, 
            ArithmeticError, OverflowError, ZeroDivisionError):
        return "Unsafe Code Detected!"
    except Exception:
        return "Unsafe Code Detected!"
