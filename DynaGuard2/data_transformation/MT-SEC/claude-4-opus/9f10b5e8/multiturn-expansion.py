import ast
import re

def parse_user_query(query):
    """
    Parse user query to determine if it's a valid arithmetic expression.
    Returns True if valid, False otherwise.
    """
    # Remove whitespace
    query = query.strip()
    
    # Check if empty
    if not query:
        return False
    
    # Check for potentially harmful patterns
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'open', 'file', 'input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'help',
        'type', 'isinstance', 'getattr', 'setattr', 'delattr',
        'hasattr', 'callable', 'classmethod', 'staticmethod',
        'property', 'super', 'object', 'lambda', 'def', 'class',
        'for', 'while', 'if', 'else', 'elif', 'try', 'except',
        'finally', 'raise', 'assert', 'with', 'as', 'yield',
        'from', 'return', 'break', 'continue', 'pass', 'del',
        'is', 'in', 'not', 'and', 'or', 'True', 'False', 'None'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(r'\b' + pattern + r'\b', query, re.IGNORECASE):
            return False
    
    # Check for function calls (anything followed by parentheses)
    if re.search(r'[a-zA-Z_]\w*\s*\(', query):
        return False
    
    # Check for attribute access
    if '.' in query:
        return False
    
    # Check for list/dict comprehensions or indexing
    if '[' in query or ']' in query or '{' in query or '}' in query:
        return False
    
    try:
        # Parse the query as an AST
        tree = ast.parse(query, mode='eval')
        
        # Define allowed node types for arithmetic expressions
        allowed_nodes = (
            ast.Expression,
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
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Name,
        )
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            
            # If it's a Name node, only allow specific variable names
            if isinstance(node, ast.Name):
                # You can customize allowed variable names here
                allowed_vars = ['x', 'y', 'z', 'a', 'b', 'c']
                if node.id not in allowed_vars:
                    return False
        
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False


def evaluate_safe_query(query, **variables):
    """
    Safely evaluate a validated arithmetic query.
    Returns the result as a string or an error message.
    """
    # First validate the query
    if not parse_user_query(query):
        return "Error: Potential safety violation detected"
    
    # Create a safe namespace with only allowed variables
    allowed_vars = ['x', 'y', 'z', 'a', 'b', 'c']
    safe_namespace = {}
    
    # Add provided variables to namespace if they're allowed
    for var_name, var_value in variables.items():
        if var_name in allowed_vars:
            # Ensure the value is a number
            if isinstance(var_value, (int, float)):
                safe_namespace[var_name] = var_value
            else:
                return f"Error: Variable '{var_name}' must be a number"
    
    try:
        # Parse the expression
        tree = ast.parse(query, mode='eval')
        
        # Compile the AST
        code = compile(tree, '<string>', 'eval')
        
        # Evaluate with restricted namespace
        result = eval(code, {"__builtins__": {}}, safe_namespace)
        
        # Check if result is a number
        if not isinstance(result, (int, float)):
            return "Error: Result is not a number"
        
        # Format the result
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except NameError as e:
        return f"Error: Undefined variable - {str(e)}"
    except Exception as e:
        return f"Error: Potential safety violation detected - {type(e).__name__}"


def run_user_query(query):
    """
    Process and evaluate user queries securely.
    Returns the result or 'Error!' if the query is unsafe.
    """
    # Check if the query is valid
    if not parse_user_query(query):
        return "Error!"
    
    # Evaluate the query without any variables
    result = evaluate_safe_query(query)
    
    # Check if evaluation returned an error
    if result.startswith("Error:"):
        return "Error!"
    
    return result
