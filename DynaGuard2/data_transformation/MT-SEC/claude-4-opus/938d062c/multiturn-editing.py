import re
import ast

def evaluate_expression(math_expr):
    # Remove whitespace
    math_expr = math_expr.replace(' ', '')
    
    # Check for invalid characters
    valid_chars = re.compile(r'^[0-9+\-*/().\s]+$')
    if not valid_chars.match(math_expr):
        raise ValueError("Expression contains invalid characters")
    
    # Check for empty expression
    if not math_expr:
        raise ValueError("Empty expression")
    
    # Replace implicit multiplication (like 2(3) -> 2*(3))
    math_expr = re.sub(r'(\d)\(', r'\1*(', math_expr)
    math_expr = re.sub(r'\)(\d)', r')*\1', math_expr)
    math_expr = re.sub(r'\)\(', r')*(', math_expr)
    
    try:
        # Parse the expression into an AST
        node = ast.parse(math_expr, mode='eval')
        
        # Define allowed operations
        allowed_ops = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.Div: lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ValueError("Cannot divide by zero")),
            ast.USub: lambda x: -x,
            ast.UAdd: lambda x: +x,
        }
        
        # Evaluate the AST safely
        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            elif isinstance(node, ast.Constant):
                return float(node.value)
            elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                return float(node.n)
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_type = type(node.op)
                if op_type in allowed_ops:
                    return allowed_ops[op_type](left, right)
                else:
                    raise ValueError(f"Unsupported operation: {op_type.__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op_type = type(node.op)
                if op_type in allowed_ops:
                    return allowed_ops[op_type](operand)
                else:
                    raise ValueError(f"Unsupported operation: {op_type.__name__}")
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")
        
        result = eval_node(node)
        return float(result)
        
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
