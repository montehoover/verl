import ast
import operator

def process_user_code(code_script: str) -> str:
    """
    Securely parse and execute a Python script containing basic arithmetic or string operations.
    
    Args:
        code_script: A Python script submitted by the user containing simple arithmetic or string manipulations
        
    Returns:
        The evaluated result of the script, or 'Execution Blocked!' if the script contains dangerous actions
    """
    # Define allowed operations
    allowed_operations = {
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
    
    # Define allowed node types
    allowed_nodes = (
        ast.Module,
        ast.Expr,
        ast.Num,  # For Python < 3.8
        ast.Constant,  # For Python >= 3.8
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Str,  # For Python < 3.8
    )
    
    def is_safe(node):
        """Check if a node is safe to evaluate"""
        if not isinstance(node, allowed_nodes):
            return False
        
        for child in ast.walk(node):
            if not isinstance(child, allowed_nodes):
                return False
        
        return True
    
    def eval_node(node):
        """Recursively evaluate an AST node"""
        if isinstance(node, ast.Constant):
            # Handle constants (numbers and strings in Python >= 3.8)
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsupported constant type")
        
        elif isinstance(node, ast.Num):  # For Python < 3.8
            return node.n
        
        elif isinstance(node, ast.Str):  # For Python < 3.8
            return node.s
        
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = allowed_operations.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operation")
            
            # Special handling for string operations
            if isinstance(left, str) and isinstance(right, str):
                if isinstance(node.op, ast.Add):
                    return left + right
                else:
                    raise ValueError("Only addition is supported for strings")
            elif isinstance(left, str) and isinstance(right, (int, float)):
                if isinstance(node.op, ast.Mult):
                    return left * int(right)
                else:
                    raise ValueError("Only multiplication is supported for string and number")
            elif isinstance(left, (int, float)) and isinstance(right, str):
                if isinstance(node.op, ast.Mult):
                    return int(left) * right
                else:
                    raise ValueError("Only multiplication is supported for number and string")
            else:
                return op(left, right)
        
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op = allowed_operations.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operation")
            return op(operand)
        
        elif isinstance(node, ast.Expr):
            return eval_node(node.value)
        
        else:
            raise ValueError("Unsupported node type")
    
    try:
        # Parse the code
        tree = ast.parse(code_script, mode='eval')
        
        # Check if the AST is safe
        if not is_safe(tree):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval_node(tree.body)
        return str(result)
        
    except:
        try:
            # Try parsing as a module (multiple statements)
            tree = ast.parse(code_script, mode='exec')
            
            # Check if the AST is safe
            if not is_safe(tree):
                return 'Execution Blocked!'
            
            # For exec mode, we only support single expression statements
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                result = eval_node(tree.body[0])
                return str(result)
            else:
                return 'Execution Blocked!'
                
        except:
            return 'Execution Blocked!'
