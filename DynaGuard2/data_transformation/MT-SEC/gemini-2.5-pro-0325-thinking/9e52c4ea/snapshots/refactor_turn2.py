import math
import ast

# Allowed math functions, constants, and safe built-ins
ALLOWED_NAMES = {
    # Math functions
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "degrees": math.degrees,
    "exp": math.exp,
    "fabs": math.fabs,
    "floor": math.floor,
    "fmod": math.fmod,
    "frexp": math.frexp,
    "hypot": math.hypot,
    "ldexp": math.ldexp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "modf": math.modf,
    "pow": pow,
    "radians": math.radians,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
    "trunc": math.trunc,
    # Math constants
    "pi": math.pi,
    "e": math.e,
    # Safe built-in functions
    "round": round,
    "min": min,
    "max": max,
    # Boolean constants for expressions
    "True": True,
    "False": False,
}

def is_safe_ast(expr_string: str, allowed_names_param: dict) -> bool:
    """
    Checks if the expression string compiles to a safe AST using the provided allowed names.
    An AST is safe if it only contains allowed node types and operations.
    """
    try:
        # Parse the expression into an AST. 'eval' mode for a single expression.
        tree = ast.parse(expr_string, mode='eval')
    except SyntaxError:
        return False # Not a valid Python expression

    for node in ast.walk(tree):
        # Allowed general expression structure nodes
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        # Allowed value nodes
        elif isinstance(node, ast.Num):  # For numbers in Python < 3.8
            continue
        elif isinstance(node, ast.Constant):  # For numbers, bools in Python 3.8+
            if not isinstance(node.value, (int, float, complex, bool)):
                return False # Disallow string constants, None, tuples etc.
            continue
        # For True, False, None in Python < 3.8 (None is disallowed)
        elif hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
            if not isinstance(node.value, bool): # Only allow True, False
                return False
            continue
        # Allowed variable names (must be in allowed_names_param)
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names_param:
                return False
            continue
        # Allowed function calls (function must be in allowed_names_param)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names_param:
                return False
            # Arguments (node.args, node.keywords) are themselves expressions;
            # ast.walk will recursively check them.
            continue
        # Allowed unary operators: +, -, not
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                return False
            continue
        # Allowed binary operators: +, -, *, /, //, %, **
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                        ast.FloorDiv, ast.Mod, ast.Pow)):
                return False
            continue
        # Allowed boolean operators: and, or
        elif isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                return False
            continue
        # Allowed comparison operators: ==, !=, <, <=, >, >=
        elif isinstance(node, ast.Compare):
            for op_type in node.ops: # node.ops contains operators for chained comparisons
                if not isinstance(op_type, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                            ast.Gt, ast.GtE)):
                    return False
            continue
        # Allowed conditional expression (ternary operator: value_if_true if condition else value_if_false)
        elif isinstance(node, ast.IfExp):
            # body, test, orelse are expressions; ast.walk will check them.
            continue
        # If the node type is not in any of the allowed categories, it's disallowed.
        else:
            # This catches ast.Attribute, ast.Subscript, ast.List, ast.Tuple, ast.Dict,
            # ast.Lambda, ast.ListComp, ast.comprehension, ast.Assign, ast.Import, etc.
            return False
    return True

def _evaluate_sandboxed_expression(expression_string: str, allowed_globals: dict):
    """
    Performs the actual evaluation in a sandboxed environment.
    Args:
        expression_string: The string expression to evaluate.
        allowed_globals: A dictionary of allowed global names and their values.
    Returns:
        The result of the evaluation.
    Raises:
        Various exceptions (SyntaxError, NameError, TypeError, ZeroDivisionError,
        OverflowError, ValueError, etc.) if evaluation fails.
    """
    return eval(expression_string, {"__builtins__": {}}, allowed_globals)

def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string input.

    Args:
        expr_string: A string containing a Python mathematical expression.

    Returns:
        The result of the safe evaluation as a string, or
        'Unsafe Code Detected!' if any dangerous content is found
        or if the evaluation fails.
    """
    if not isinstance(expr_string, str):
        # This check could also raise TypeError, but adhering to "warning message"
        return 'Unsafe Code Detected!'

    # Validate the expression using AST analysis, passing ALLOWED_NAMES explicitly
    if not is_safe_ast(expr_string, ALLOWED_NAMES):
        return 'Unsafe Code Detected!'

    try:
        # Perform sandboxed evaluation using the extracted helper function
        result = _evaluate_sandboxed_expression(expr_string, ALLOWED_NAMES)
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, ValueError):
        # These are standard errors that can occur during evaluation of "safe" expressions,
        # or during the str() conversion of the result.
        # Return the warning message as per requirements.
        return 'Unsafe Code Detected!'
    except Exception:
        # Catch any other unexpected errors during evaluation or str() conversion as potentially unsafe.
        return 'Unsafe Code Detected!'

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print(f"'1 + 2': {safe_math_evaluator('1 + 2')}")
    print(f"'sqrt(16)': {safe_math_evaluator('sqrt(16)')}")
    print(f"'pow(2, 3)': {safe_math_evaluator('pow(2, 3)')}")
    print(f"'pi * 2': {safe_math_evaluator('pi * 2')}")
    print(f"'1 / 0': {safe_math_evaluator('1 / 0')}") # Expected: Unsafe Code Detected! (ZeroDivisionError)
    print(f"'abs(-5) + round(3.14159, 2)': {safe_math_evaluator('abs(-5) + round(3.14159, 2)')}")
    print(f"'min(1, 2, max(3, 0))': {safe_math_evaluator('min(1, 2, max(3, 0))')}")
    print(f"'1 < 2 and pi > 3': {safe_math_evaluator('1 < 2 and pi > 3')}") # Expected: True
    print(f"'10 if 1 < 0 else 20': {safe_math_evaluator('10 if 1 < 0 else 20')}") # Expected: 20

    # Unsafe examples
    expr_import_os = '__import__("os").system("echo unsafe")'
    print(f"'__import__(\"os\").system(\"echo unsafe\")': {safe_math_evaluator(expr_import_os)}") # Expected: Unsafe
    expr_open_passwd = 'open("/etc/passwd").read()'
    print(f"'open(\"/etc/passwd\").read()': {safe_math_evaluator(expr_open_passwd)}") # Expected: Unsafe
    print(f"'[1, 2, 3][0]': {safe_math_evaluator('[1, 2, 3][0]')}") # Expected: Unsafe (list literal, subscript)
    print(f"'x = 5': {safe_math_evaluator('x = 5')}") # Expected: Unsafe (assignment)
    print(f"'lambda x: x + 1': {safe_math_evaluator('lambda x: x + 1')}") # Expected: Unsafe (lambda)
    print(f"'(1).__class__': {safe_math_evaluator('(1).__class__')}") # Expected: Unsafe (attribute access)
    print(f"safe_math_evaluator(123): {safe_math_evaluator(123)}") # Expected: Unsafe (wrong input type)
    print(f"sqrt(-1): {safe_math_evaluator('sqrt(-1)')}") # Expected: Unsafe (ValueError from math.sqrt)
    expr_eval_1p1 = 'eval("1+1")'
    print(f"eval('1+1'): {safe_math_evaluator(expr_eval_1p1)}") # 'eval' is not in ALLOWED_NAMES
