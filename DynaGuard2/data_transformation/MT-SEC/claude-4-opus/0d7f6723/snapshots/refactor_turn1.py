import ast
import operator

def parse_user_script(script_str: str) -> str:
    """
    Safely evaluate a user-provided Python script containing only simple arithmetic and string operations.
    
    Args:
        script_str: A string containing a tiny Python script with only basic arithmetic or string operations
        
    Returns:
        Either the result of the script execution or a security warning if the input is unsafe
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
    
    # Define allowed node types
    allowed_nodes = (
        ast.Module,
        ast.Expr,
        ast.Constant,
        ast.Num,  # For Python < 3.8 compatibility
        ast.Str,  # For Python < 3.8 compatibility
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
    )
    
    def evaluate_node(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Only allow numbers and strings
            if isinstance(node.value, (int, float, str)):
                return node.value
            else:
                raise ValueError("Unsafe constant type")
                
        # For Python < 3.8 compatibility
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
            
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            
            # Get the operator function
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError("Unsafe operator")
                
            op_func = allowed_operators[op_type]
            
            # Handle string concatenation
            if isinstance(left, str) and isinstance(right, str) and op_type == ast.Add:
                return left + right
            # Handle string repetition
            elif isinstance(left, str) and isinstance(right, int) and op_type == ast.Mult:
                return left * right
            elif isinstance(left, int) and isinstance(right, str) and op_type == ast.Mult:
                return left * right
            # Handle numeric operations
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                try:
                    return op_func(left, right)
                except ZeroDivisionError:
                    raise ValueError("Division by zero")
            else:
                raise ValueError("Invalid operation")
                
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            
            # Get the operator function
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError("Unsafe operator")
                
            op_func = allowed_operators[op_type]
            
            # Only allow unary operations on numbers
            if isinstance(operand, (int, float)):
                return op_func(operand)
            else:
                raise ValueError("Invalid unary operation")
                
        else:
            raise ValueError("Unsafe node type")
    
    try:
        # Parse the script
        tree = ast.parse(script_str, mode='eval')
        
        # Check if all nodes are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "Execution Blocked!"
        
        # Evaluate the expression
        result = evaluate_node(tree.body)
        
        # Convert result to string
        return str(result)
        
    except:
        return "Execution Blocked!"
