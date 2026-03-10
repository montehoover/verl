import math
import ast
import re

def process_user_expression(expr: str) -> str:
    """
    Evaluates a user-provided mathematical expression safely.

    Args:
        expr: A string containing the user's expression.

    Returns:
        The result of the evaluated expression as a string,
        or 'Processing Failed!' if the input is invalid,
        raises an unhandled exception, or a security risk is detected.
    """
    # Maximum length for an expression
    if len(expr) > 200: # Arbitrary limit, adjust as needed
        return 'Processing Failed!'

    # Deny list for keywords that are definitely unsafe even if not Python keywords
    # This is a secondary check. The primary check is AST parsing.
    unsafe_substrings = [
        "__import__", "eval", "exec", "open", "compile",
        "getattr", "setattr", "delattr", "globals", "locals",
        "os", "sys", "subprocess", "shutil", "socket", "urllib", "requests",
        "lambda", "input", "type", "dir", "help", "breakpoint",
        "class", "import", "from", # Python keywords that are dangerous in this context
        "_", # common way to access last result, can be problematic if it holds arbitrary objects
        # Patterns that might indicate attempts to access dunder methods or attributes
        ".__", "__.", "[", "]", "{", "}" # Brackets/braces are disallowed by AST check for Name nodes
    ]

    expr_lower = expr.lower()
    for keyword in unsafe_substrings:
        if keyword in expr_lower:
            return 'Processing Failed!'

    # Whitelist of allowed functions and constants
    allowed_names = {
        'abs': abs,
        'round': round,
        'pow': pow,
        'min': min,
        'max': max,
        # math functions
        'ceil': math.ceil,
        'floor': math.floor,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'degrees': math.degrees,
        'radians': math.radians,
        # math constants
        'pi': math.pi,
        'e': math.e,
    }

    try:
        # 1. Parse the expression into an Abstract Syntax Tree (AST)
        parsed_expr = ast.parse(expr, mode='eval')

        # 2. Validate the AST nodes
        #    - Only allow specific node types (e.g., Name, Num, BinOp, UnaryOp, Call, Constant for Python 3.8+)
        #    - For Name nodes, ensure they are in our allowed_names whitelist.
        #    - For Call nodes, ensure the function being called is in our allowed_names whitelist.
        #    - Disallow attribute access (ast.Attribute).
        #    - Disallow subscriptions/slicing (ast.Subscript).
        #    - Disallow comprehensions, lambdas, etc.

        allowed_node_types = (
            ast.Expression, ast.Call, ast.Name, ast.Load,
            ast.BinOp, ast.UnaryOp, ast.Num, # ast.Num is for < Python 3.8
            ast.Constant, # ast.Constant is for >= Python 3.8 (includes numbers, None, True, False)
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, # Binary operators
            ast.USub, ast.UAdd # Unary operators
        )

        for node in ast.walk(parsed_expr):
            if not isinstance(node, allowed_node_types):
                # print(f"Disallowed node type: {type(node)}") # For debugging
                return 'Processing Failed!'
            
            if isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    # print(f"Disallowed name: {node.id}") # For debugging
                    return 'Processing Failed!'
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                    # print(f"Disallowed function call: {ast.dump(node.func)}") # For debugging
                    return 'Processing Failed!'
                # Ensure all arguments are also valid expressions
                for arg_node in node.args:
                    if not isinstance(arg_node, (ast.Num, ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Call)):
                        return 'Processing Failed!'
                for kwarg_node in node.keywords: # Disallow keyword arguments for simplicity
                     return 'Processing Failed!'

            elif isinstance(node, ast.Attribute): # Explicitly disallow attribute access
                return 'Processing Failed!'


        # 3. If AST is valid, evaluate the expression.
        #    The `eval` environment will only contain the whitelisted names.
        #    `__builtins__` is empty to prevent access to any built-in functions not explicitly allowed.
        
        # Create a safe dictionary for eval's globals
        safe_globals = {"__builtins__": {}}
        safe_globals.update(allowed_names)
        
        result = eval(compile(parsed_expr, filename='<user_expr>', mode='eval'), safe_globals, {})

        # Ensure the result is a number (int, float, complex)
        if not isinstance(result, (int, float, complex)):
            # print(f"Unexpected result type: {type(result)}") # For debugging
            return 'Processing Failed!'
            
        return str(result)

    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, ValueError, AttributeError) as e:
        # print(f"Evaluation error: {e}") # For debugging
        return 'Processing Failed!'
    except RecursionError: # AST walking or eval could hit recursion limits for complex expressions
        return 'Processing Failed!'
    except MemoryError:
        return 'Processing Failed!'
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        # print(f"Unexpected exception: {e}") # For debugging
        return 'Processing Failed!'
