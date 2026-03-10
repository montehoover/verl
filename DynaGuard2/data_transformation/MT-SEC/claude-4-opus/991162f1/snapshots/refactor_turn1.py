import ast
import operator
import math

def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluate a mathematical expression string without using eval() or exec().
    
    Args:
        math_expression: A string representing a mathematical expression in Python
        
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
        'sum': sum,
        'len': len,
        'int': int,
        'float': float,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def evaluate_node(node):
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Unsafe Code Detected!")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operators:
                raise ValueError("Unsafe Code Detected!")
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operators:
                raise ValueError("Unsafe Code Detected!")
            operand = evaluate_node(node.operand)
            return allowed_operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in allowed_functions:
                    raise ValueError("Unsafe Code Detected!")
                args = [evaluate_node(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            else:
                raise ValueError("Unsafe Code Detected!")
        elif isinstance(node, ast.Name):
            if node.id in allowed_functions and isinstance(allowed_functions[node.id], (int, float)):
                return allowed_functions[node.id]
            else:
                raise ValueError("Unsafe Code Detected!")
        elif isinstance(node, ast.List):
            return [evaluate_node(elem) for elem in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(evaluate_node(elem) for elem in node.elts)
        else:
            raise ValueError("Unsafe Code Detected!")
    
    try:
        # Parse the expression
        tree = ast.parse(math_expression, mode='eval')
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Return the result as a string
        return str(result)
        
    except:
        return "Unsafe Code Detected!"
