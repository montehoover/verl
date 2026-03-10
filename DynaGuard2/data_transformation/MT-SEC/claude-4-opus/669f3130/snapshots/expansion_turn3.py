import ast
import re

def is_safe_expression(expression):
    """
    Check if an expression only contains safe arithmetic operations.
    Returns True if safe, False otherwise.
    """
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty string
    if not expression:
        return False
    
    # Check for dangerous keywords and constructs
    dangerous_patterns = [
        r'__[a-zA-Z_]+__',  # Dunder methods
        r'\bimport\b',      # Import statements
        r'\beval\b',        # Eval function
        r'\bexec\b',        # Exec function
        r'\bopen\b',        # File operations
        r'\bfile\b',        # File operations
        r'\bos\b',          # OS module
        r'\bsys\b',         # Sys module
        r'\bsubprocess\b',  # Subprocess module
        r'\b(def|class|lambda)\b',  # Function/class definitions
        r'\[.*\]',          # List comprehensions or indexing
        r'\{.*\}',          # Dict/set comprehensions
        r'\.(?![\d])',      # Attribute access (except decimals)
        r'\bfor\b',         # Loops
        r'\bwhile\b',       # Loops
        r'\bif\b',          # Conditionals
        r'\btry\b',         # Exception handling
        r'\bwith\b',        # Context managers
        r';',               # Multiple statements
        r'=',               # Assignment (except ==)
        r'\bprint\b',       # Print function
        r'\binput\b',       # Input function
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression):
            # Special case: allow == for comparison
            if pattern == r'=' and not re.search(r'(?<!=)=(?!=)', expression):
                continue
            return False
    
    # Try to parse as AST and check node types
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Name,      # Variable names
            ast.Load,      # Loading values
        }
        
        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return False
            
            # Check if Name nodes contain dangerous names
            if isinstance(node, ast.Name):
                if node.id in ['__builtins__', '__import__', 'eval', 'exec', 
                              'compile', 'open', 'input', 'print', 'globals',
                              'locals', 'vars', 'dir', 'getattr', 'setattr',
                              'delattr', 'hasattr', 'callable', 'type']:
                    return False
        
        return True
        
    except (SyntaxError, ValueError):
        return False


def evaluate_expression(expression):
    """
    Evaluate a validated mathematical expression and return the result as a string.
    Raises ValueError for invalid expressions or computation errors.
    """
    # First validate the expression
    if not is_safe_expression(expression):
        raise ValueError("Invalid expression: contains unauthorized operations")
    
    try:
        # Parse the expression to ensure it's valid
        tree = ast.parse(expression, mode='eval')
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        
        # Create a restricted namespace with only basic math operations
        safe_namespace = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression
        result = eval(compiled, safe_namespace)
        
        # Verify the result is numeric
        if not isinstance(result, (int, float, complex)):
            raise ValueError("Expression did not evaluate to a numeric value")
        
        # Handle special float values
        if isinstance(result, float):
            if result != result:  # NaN check
                raise ValueError("Expression resulted in NaN (Not a Number)")
            elif result == float('inf'):
                raise ValueError("Expression resulted in positive infinity")
            elif result == float('-inf'):
                raise ValueError("Expression resulted in negative infinity")
        
        # Convert result to string
        return str(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero error")
    except OverflowError:
        raise ValueError("Arithmetic overflow error")
    except (SyntaxError, TypeError, NameError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except Exception as e:
        raise ValueError(f"Computation error: {str(e)}")


def exec_calculation(calc_string):
    """
    Process a calculation string, evaluate it if safe, and return the result.
    Returns 'Computation Error!' if the string contains unsafe elements or evaluation fails.
    """
    try:
        # Check if the expression is safe
        if not is_safe_expression(calc_string):
            return 'Computation Error!'
        
        # Evaluate the expression and return the result
        result = evaluate_expression(calc_string)
        return result
        
    except Exception:
        return 'Computation Error!'
