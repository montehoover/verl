import ast
import operator

def evaluate_user_math(math_expr: str) -> str:
    try:
        # Define allowed operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.And: lambda x, y: x and y,
            ast.Or: lambda x, y: x or y,
            ast.Not: lambda x: not x,
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # For Python < 3.8
                return node.n
            elif isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                return operators[type(node.op)](safe_eval(node.operand))
            elif isinstance(node, ast.BinOp) and type(node.op) in operators:
                return operators[type(node.op)](safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.BoolOp) and type(node.op) in operators:
                values = [safe_eval(v) for v in node.values]
                if isinstance(node.op, ast.And):
                    result = values[0]
                    for v in values[1:]:
                        result = result and v
                    return result
                elif isinstance(node.op, ast.Or):
                    result = values[0]
                    for v in values[1:]:
                        result = result or v
                    return result
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not safe_eval(node.operand)
            elif isinstance(node, ast.Compare):
                left = safe_eval(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    right = safe_eval(comparator)
                    if isinstance(op, ast.Eq):
                        if not (left == right):
                            return False
                    elif isinstance(op, ast.NotEq):
                        if not (left != right):
                            return False
                    elif isinstance(op, ast.Lt):
                        if not (left < right):
                            return False
                    elif isinstance(op, ast.LtE):
                        if not (left <= right):
                            return False
                    elif isinstance(op, ast.Gt):
                        if not (left > right):
                            return False
                    elif isinstance(op, ast.GtE):
                        if not (left >= right):
                            return False
                    else:
                        raise ValueError("Unsupported operation")
                    left = right
                return True
            else:
                raise ValueError("Unsupported operation")
        
        # Parse the expression
        tree = ast.parse(math_expr, mode='eval')
        
        # Evaluate safely
        result = safe_eval(tree.body)
        
        # Return result as string
        return str(result)
        
    except:
        return "Processing Failed!"
