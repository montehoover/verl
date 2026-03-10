import ast
import re

def filter_unsafe_operations(script):
    """
    Check if a Python script contains only safe operations (arithmetic and string manipulations).
    Returns True if safe, False otherwise.
    """
    # List of allowed AST node types for safe operations
    allowed_nodes = {
        # Literals
        ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant,
        
        # Basic operations
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        
        # Arithmetic operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult,
        
        # Unary operators
        ast.Invert, ast.Not, ast.UAdd, ast.USub,
        
        # Comparison operators
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
        ast.In, ast.NotIn,
        
        # Boolean operators
        ast.And, ast.Or,
        
        # Basic containers (but no comprehensions)
        ast.List, ast.Tuple, ast.Set,
        
        # String formatting
        ast.JoinedStr, ast.FormattedValue,
        
        # Basic statements
        ast.Expr, ast.Assign, ast.AugAssign,
        
        # Names (variables)
        ast.Name, ast.Store, ast.Load, ast.Del,
        
        # Subscripting for basic indexing
        ast.Subscript, ast.Index, ast.Slice,
        
        # Module level
        ast.Module,
    }
    
    # List of allowed built-in names for safe operations
    allowed_builtins = {
        'True', 'False', 'None',
        'int', 'float', 'str', 'bool',
        'len', 'min', 'max', 'sum', 'abs', 'round',
        'sorted', 'reversed',
    }
    
    try:
        # Parse the script into an AST
        tree = ast.parse(script)
        
        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            # Check if node type is allowed
            if type(node) not in allowed_nodes:
                return False
            
            # Additional checks for specific node types
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Check if the name being loaded is an allowed builtin
                if node.id not in allowed_builtins and not node.id.isidentifier():
                    return False
            
            # Prevent attribute access (no dots allowed)
            if isinstance(node, ast.Attribute):
                return False
            
            # Prevent function calls except for allowed builtins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_builtins:
                        return False
                else:
                    return False
            
            # Prevent imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            
            # Prevent function/class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            
            # Prevent loops and control flow (except basic if)
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                return False
            
            # Prevent exception handling
            if isinstance(node, (ast.Try, ast.Raise, ast.Assert)):
                return False
            
            # Prevent generators and comprehensions
            if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
                return False
            
            # Prevent yield/return
            if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Return)):
                return False
            
            # Prevent global/nonlocal
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False
            
            # Prevent exec/eval via string
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in ('exec', 'eval', 'compile', '__import__'):
                        return False
        
        return True
        
    except SyntaxError:
        # If the script has syntax errors, it's not safe
        return False
    except:
        # Any other parsing errors mean it's not safe
        return False
