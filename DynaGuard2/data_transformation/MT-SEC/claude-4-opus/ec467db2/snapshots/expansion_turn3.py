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


def execute_safe_script(script):
    """
    Execute a script that has been verified as safe.
    Returns the result of the operations or an error message.
    """
    # Create a restricted namespace with only allowed builtins
    allowed_builtins = {
        'True': True,
        'False': False,
        'None': None,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'len': len,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'reversed': reversed,
        '__builtins__': {}
    }
    
    # Create a local namespace for script execution
    local_namespace = {}
    
    try:
        # Compile and execute the script
        compiled_code = compile(script, '<safe_script>', 'exec')
        exec(compiled_code, allowed_builtins, local_namespace)
        
        # Check if there's a single expression to return
        try:
            # Try to evaluate as an expression to get a return value
            result = eval(compile(script, '<safe_script>', 'eval'), allowed_builtins, local_namespace)
            return result
        except:
            # If not a single expression, check for variables in local namespace
            if local_namespace:
                # Return the last assigned variable or all variables
                if len(local_namespace) == 1:
                    return list(local_namespace.values())[0]
                else:
                    return local_namespace
            else:
                # No result to return
                return None
                
    except ZeroDivisionError:
        return "Error: Division by zero encountered in the script"
    except ValueError as e:
        return f"Error: Value error in the script - {str(e)}"
    except TypeError as e:
        return f"Error: Type error in the script - {str(e)}"
    except MemoryError:
        return "Error: Memory error - the script uses too much memory"
    except RecursionError:
        return "Error: Recursion error - the script has too many recursive calls"
    except Exception as e:
        return f"Error: Problem executing the script - {str(e)}"


def execute_user_code(user_code):
    """
    Process user_code to securely parse and execute the script.
    Returns 'Execution Blocked!' if unsafe operations are detected,
    otherwise returns the result of execution.
    """
    # First check if the code is safe
    if not filter_unsafe_operations(user_code):
        return 'Execution Blocked!'
    
    # If safe, execute the script
    return execute_safe_script(user_code)
