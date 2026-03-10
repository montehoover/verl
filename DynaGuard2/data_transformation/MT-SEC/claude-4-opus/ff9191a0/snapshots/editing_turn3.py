import re
import operator
import ast

def evaluate_expression_safely(math_expression, variable_mapping):
    """
    Securely evaluate mathematical expressions with variables.
    
    Args:
        math_expression: A string of the expression
        variable_mapping: A dictionary mapping variable names to numbers
        
    Returns:
        The result as a string
        
    Raises:
        ValueError: If the input expression is invalid or computation fails
    """
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
                raise ValueError("Invalid constant in expression")
        elif isinstance(node, ast.Name):
            if node.id in variable_mapping:
                value = variable_mapping[node.id]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Variable '{node.id}' must be numeric")
                return value
            else:
                raise ValueError(f"Variable '{node.id}' is not defined in the provided mapping")
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            op_type = type(node.op)
            if op_type in operators:
                try:
                    return operators[op_type](left, right)
                except ZeroDivisionError:
                    raise ValueError("Division by zero")
            else:
                raise ValueError("Unsupported operation")
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            else:
                raise ValueError("Unsupported operation")
        else:
            raise ValueError("Invalid expression")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(math_expression, mode='eval')
        
        # Evaluate the expression tree
        result = evaluate_node(tree.body)
        
        # Return result as string
        return str(result)
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Computation failed: {str(e)}")
