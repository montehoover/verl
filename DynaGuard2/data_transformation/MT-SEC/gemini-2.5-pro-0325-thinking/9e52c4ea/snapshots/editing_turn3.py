import math
import ast

# Allowed math functions and constants
ALLOWED_NAMES = {
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
    "log1p": math.log1p,
    "pow": pow, # Using built-in pow for consistency with **
    "radians": math.radians,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
    "pi": math.pi,
    "e": math.e,
    # You can add more functions or constants here if needed
}

# Restricted builtins: only allow what's absolutely necessary for evaluation
# or what's explicitly whitelisted.
# For mathematical expressions, often an empty __builtins__ or a very restricted one is safest.
# Here, we provide an empty dict to prevent access to any builtins by default.
# If specific builtins like `round` are needed, they can be added to ALLOWED_NAMES.
SAFE_BUILTINS = {}

def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluates a string containing a Python mathematical expression.

    Args:
        expr_string: The mathematical expression string to evaluate.

    Returns:
        The result of the evaluation as a string, or a warning message
        if the expression is invalid or poses a security risk.
    """
    if not isinstance(expr_string, str):
        return "Warning: Invalid input type. Expression must be a string."

    # Preliminary check for obviously malicious characters or patterns
    # This is a basic check; the primary safety comes from AST parsing and controlled eval environment.
    # Disallow dunder methods, import statements, etc.
    if "__" in expr_string or "import" in expr_string or "lambda" in expr_string or "eval" in expr_string or "exec" in expr_string:
        return "Warning: Expression contains potentially unsafe constructs."

    try:
        # Parse the expression into an AST (Abstract Syntax Tree)
        # This allows us to inspect the structure before evaluation.
        node = ast.parse(expr_string, mode='eval')

        # Validate nodes in the AST to ensure only allowed operations/names are used
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Name) and sub_node.id not in ALLOWED_NAMES:
                # Check if it's a number (which ast.Name doesn't cover directly for literals)
                # This case is more for variables; numbers are ast.Num (Python 3.7) or ast.Constant (Python 3.8+)
                try:
                    # Attempt to convert to float to see if it's a numeric literal not in ALLOWED_NAMES
                    # This is a bit redundant as ast.Num/Constant handles numbers.
                    # The main check here is for non-allowed variable names.
                    float(sub_node.id) 
                except ValueError:
                     return f"Warning: Use of undefined or disallowed name '{sub_node.id}'."
            elif isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name) and sub_node.func.id not in ALLOWED_NAMES:
                    return f"Warning: Use of disallowed function '{sub_node.func.id}'."
                # Further checks for attributes or complex calls can be added if needed
            elif not isinstance(sub_node, (
                ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, # Python 3.8+
                ast.Num, ast.Str, # Python 3.7 and earlier (Num for numbers, Str for strings)
                ast.BinOp, ast.UnaryOp, ast.Compare, ast.operator,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, # Binary operators
                ast.UAdd, ast.USub, ast.Not, ast.Invert, # Unary operators
                ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, # Compare operators
                ast.Tuple, ast.List, # If you want to allow tuples/lists as part of expressions
                ast.Slice # If you want to allow slicing (less common for pure math)
            )):
                return f"Warning: Use of disallowed language construct '{type(sub_node).__name__}'."


        # Compile the AST node into a code object.
        # The <string> argument is a filename placeholder for error messages.
        code = compile(node, '<string>', 'eval')

        # Evaluate the compiled code object in a restricted environment.
        # Globals dict provides allowed names (math functions, constants).
        # Locals dict can be the same or a separate dict if needed.
        # __builtins__ is explicitly restricted.
        result = eval(code, {"__builtins__": SAFE_BUILTINS}, ALLOWED_NAMES)

        if not isinstance(result, (int, float, complex)): # Allow complex numbers as well
             return "Warning: Evaluation resulted in a non-numeric type."
        
        return str(result)

    except SyntaxError:
        return "Warning: Invalid syntax in expression."
    except NameError as e:
        # This might be caught by AST check, but good as a fallback.
        return f"Warning: Undefined variable or function: {e}."
    except TypeError as e:
        return f"Warning: Type error in expression: {e}."
    except ZeroDivisionError:
        return "Warning: Division by zero."
    except OverflowError:
        return "Warning: Numerical result out of range."
    except MemoryError:
        return "Warning: Expression too complex or large, causing memory issues."
    except Exception as e:
        # Catch-all for any other unexpected errors during parsing or evaluation.
        return f"Warning: An unexpected error occurred: {type(e).__name__} - {e}."

if __name__ == '__main__':
    # Test cases
    print(f"Expression '2 + 3 * 4': {safe_math_evaluator('2 + 3 * 4')}")  # Expected: 14
    print(f"Expression '(2 + 3) * 4': {safe_math_evaluator('(2 + 3) * 4')}") # Expected: 20
    print(f"Expression '2 ** 3': {safe_math_evaluator('2 ** 3')}") # Expected: 8
    print(f"Expression 'sqrt(16)': {safe_math_evaluator('sqrt(16)')}") # Expected: 4.0
    print(f"Expression 'pi * 2': {safe_math_evaluator('pi * 2')}") # Expected: ~6.28
    print(f"Expression '1 / 0': {safe_math_evaluator('1 / 0')}") # Expected: Warning: Division by zero.
    print(f"Expression '1 +': {safe_math_evaluator('1 +')}") # Expected: Warning: Invalid syntax...
    print(f"Expression 'some_var + 1': {safe_math_evaluator('some_var + 1')}") # Expected: Warning: Use of undefined...

    # Define strings for test cases that involve quotes to avoid f-string parsing issues in older Python versions
    test_expr_import_os = '__import__("os").system("echo pwned")'
    test_expr_open_file = 'open("/etc/passwd").read()'
    test_expr_eval_str = 'eval("1+1")'

    print(f"Expression '{test_expr_import_os}': {safe_math_evaluator(test_expr_import_os)}") # Expected: Warning: Expression contains potentially unsafe constructs.
    print(f"Expression '{test_expr_open_file}': {safe_math_evaluator(test_expr_open_file)}") # Expected: Warning: Use of disallowed function 'open'. (or similar based on AST check)
    print(f"Expression '{test_expr_eval_str}': {safe_math_evaluator(test_expr_eval_str)}") # Expected: Warning: Expression contains potentially unsafe constructs.

    print(f"Expression 'pow(2,3)': {safe_math_evaluator('pow(2,3)')}") # Expected: 8
    print(f"Expression 'cos(pi)': {safe_math_evaluator('cos(pi)')}") # Expected: -1.0
    print(f"Expression '1.0 / 2.0': {safe_math_evaluator('1.0 / 2.0')}") # Expected: 0.5
    print(f"Expression '1e10': {safe_math_evaluator('1e10')}") # Expected: 10000000000.0
    print(f"Expression 'max(1,2)': {safe_math_evaluator('max(1,2)')}") # Expected: Warning: Use of disallowed function 'max'.
    # Test for very large number that might cause OverflowError in eval but not during parsing
    print(f"Expression '1e308 * 10': {safe_math_evaluator('1e308 * 10')}") # Expected: Warning: Numerical result out of range.
    # Test for complex expression that might consume too much memory (hard to test reliably without specific limits)
    # Example: a very deeply nested expression or one generating huge numbers.
    # print(f"Expression '2**2**2**2**2': {safe_math_evaluator('2**2**2**2**2')}") # Might be too slow or cause OverflowError
    print(f"Expression 'for i in range(10): pass': {safe_math_evaluator('for i in range(10): pass')}") # Expected: Warning: Invalid syntax... (mode='eval' disallows statements)
    print(f"Expression '[x for x in [1,2,3]]': {safe_math_evaluator('[x for x in [1,2,3]]')}") # Expected: Warning: Use of disallowed language construct 'ListComp'.
    print(f"Expression '1 if True else 0': {safe_math_evaluator('1 if True else 0')}") # Expected: Warning: Use of disallowed language construct 'IfExp'. (or 'Name' for True if not whitelisted)
