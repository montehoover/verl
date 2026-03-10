import ast
import re

def check_script_syntax(script):
    """
    Verify if a script contains only simple arithmetic or string operations.
    Returns True if the script is safe, False otherwise.
    """
    # Check for dangerous keywords and patterns
    dangerous_patterns = [
        r'\b(import|from|exec|eval|compile|open|file|input|raw_input|__import__|globals|locals|vars|dir)\b',
        r'\b(os|sys|subprocess|socket|requests|urllib|http|ftplib|telnetlib|paramiko)\b',
        r'\b(eval|exec|compile|__builtins__|__import__|__loader__|__file__|__name__|__package__|__cached__)\b',
        r'__[a-zA-Z_]+__',  # Dunder methods
        r'\.|getattr|setattr|delattr|hasattr',  # Attribute access
        r'lambda',  # Lambda functions
        r'def\s+\w+',  # Function definitions
        r'class\s+\w+',  # Class definitions
        r'with\s+',  # Context managers
        r'try:|except:|finally:',  # Exception handling
        r'raise\s+',  # Raising exceptions
        r'assert\s+',  # Assertions
        r'yield\s+',  # Generators
        r'async\s+|await\s+',  # Async operations
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, script, re.IGNORECASE):
            return False
    
    # Try to parse as AST to check for allowed operations only
    try:
        tree = ast.parse(script, mode='exec')
        
        # Define allowed node types for simple arithmetic and string operations
        allowed_nodes = {
            ast.Module,
            ast.Expr,
            ast.Assign,
            ast.AugAssign,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.IfExp,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
            ast.Str,       # For older Python versions
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Del,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Invert,
            ast.UAdd,
            ast.USub,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
        }
        
        # Walk through the AST and check if all nodes are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return False
            
            # Additional checks for Name nodes
            if isinstance(node, ast.Name):
                # Only allow simple variable names, no builtins or special names
                if node.id.startswith('_') or node.id in ['True', 'False', 'None']:
                    if node.id not in ['True', 'False', 'None']:
                        return False
        
        return True
        
    except SyntaxError:
        return False
    except Exception:
        return False


def evaluate_safe_script(script):
    """
    Execute a validated script containing only arithmetic and string operations.
    Returns the result of execution or an error message.
    """
    # First validate the script
    if not check_script_syntax(script):
        return "Error: Security concern - script contains disallowed operations"
    
    # Create a restricted namespace with no builtins
    safe_namespace = {
        '__builtins__': {},
        'True': True,
        'False': False,
        'None': None,
    }
    
    try:
        # Parse the script
        tree = ast.parse(script, mode='exec')
        
        # Additional runtime check for safety
        for node in ast.walk(tree):
            # Check for any function calls
            if isinstance(node, ast.Call):
                return "Error: Security concern - function calls are not allowed"
            
            # Check for attribute access
            if isinstance(node, ast.Attribute):
                return "Error: Security concern - attribute access is not allowed"
            
            # Check for subscript operations
            if isinstance(node, ast.Subscript):
                return "Error: Security concern - subscript operations are not allowed"
            
            # Check for list/dict/set comprehensions
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                return "Error: Security concern - comprehensions are not allowed"
        
        # Compile and execute
        compiled_code = compile(tree, '<string>', 'exec')
        
        # Execute in the restricted namespace
        exec(compiled_code, safe_namespace)
        
        # Get the last expression value if any
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # Re-compile just the last expression to get its value
            last_expr = ast.Expression(body=tree.body[-1].value)
            expr_code = compile(last_expr, '<string>', 'eval')
            result = eval(expr_code, safe_namespace)
            return result
        
        # If no expression to return, check for variables in namespace
        # Remove the built-in keys we added
        user_vars = {k: v for k, v in safe_namespace.items() 
                     if k not in ['__builtins__', 'True', 'False', 'None']}
        
        if user_vars:
            # Return the namespace of user-defined variables
            return user_vars
        
        return None
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ArithmeticError as e:
        return f"Error: Arithmetic error - {str(e)}"
    except TypeError as e:
        return f"Error: Type error - {str(e)}"
    except NameError as e:
        return f"Error: Name error - {str(e)}"
    except Exception as e:
        return f"Error: Security concern - unexpected error: {str(e)}"
