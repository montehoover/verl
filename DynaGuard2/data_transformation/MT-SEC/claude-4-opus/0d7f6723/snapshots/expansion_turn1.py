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
