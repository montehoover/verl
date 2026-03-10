import ast
import operator

def evaluate_math_expression(math_expression):
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def evaluate_node(node):
        if isinstance(node, ast.Num):  # For Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # For Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Only numeric constants are allowed")
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            return operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
    
    try:
        # Parse the expression
        tree = ast.parse(math_expression, mode='eval')
        
        # Evaluate the expression tree
        result = evaluate_node(tree.body)
        
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, KeyError) as e:
        return f"Warning: {str(e)}"
    except Exception as e:
        return f"Warning: Invalid expression - {str(e)}"
