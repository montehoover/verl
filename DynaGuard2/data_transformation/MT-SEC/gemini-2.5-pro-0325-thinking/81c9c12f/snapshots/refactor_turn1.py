import ast
import math

# Define the whitelist of functions and constants
# This dictionary will also be used as the globals for eval
SAFE_GLOBALS = {
    # Built-in functions
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    # Math module functions
    "acos": math.acos, "acosh": math.acosh, "asin": math.asin, "asinh": math.asinh,
    "atan": math.atan, "atan2": math.atan2, "atanh": math.atanh, "ceil": math.ceil,
    "copysign": math.copysign, "cos": math.cos, "cosh": math.cosh, "degrees": math.degrees,
    "erf": math.erf, "erfc": math.erfc, "exp": math.exp, "expm1": math.expm1,
    "fabs": math.fabs, "factorial": math.factorial, "floor": math.floor, "fmod": math.fmod,
    "frexp": math.frexp, "fsum": math.fsum, "gamma": math.gamma, "gcd": math.gcd,
    "hypot": math.hypot, "isclose": math.isclose, "isfinite": math.isfinite,
    "isinf": math.isinf, "isnan": math.isnan, "ldexp": math.ldexp, "lgamma": math.lgamma,
    "log": math.log, "log10": math.log10, "log1p": math.log1p, "log2": math.log2,
    "modf": math.modf, # "pow" is duplicated from built-in, math.pow is float-only
    "radians": math.radians, "remainder": math.remainder,
    "sin": math.sin, "sinh": math.sinh, "sqrt": math.sqrt, "tan": math.tan,
    "tanh": math.tanh, "trunc": math.trunc,
    # Math module constants
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan,
    # Boolean values are handled by ast.Constant, but if accessed by Name:
    "True": True, "False": False, "None": None,
}

# Prepare the globals dictionary for eval, restricting builtins
SAFE_GLOBALS_FOR_EVAL = {"__builtins__": {}}
SAFE_GLOBALS_FOR_EVAL.update(SAFE_GLOBALS)


class SafeExpressionVisitor(ast.NodeVisitor):
    ALLOWED_NODE_TYPES = {
        ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
        ast.List, ast.Tuple,
        # Operators (these are node types for operators, not classes themselves)
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Not, ast.Invert,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.And, ast.Or,
    }

    def __init__(self):
        self.safe = True

    def visit(self, node):
        if not self.safe:  # If already found unsafe, stop visiting
            return

        if type(node) not in self.ALLOWED_NODE_TYPES:
            self.safe = False
            return

        super().visit(node)  # Continue visiting children

    def visit_Name(self, node):
        if node.id not in SAFE_GLOBALS:
            self.safe = False
            return
        self.generic_visit(node)

    def visit_Constant(self, node):
        # Allow numbers (int, float), booleans (True, False), and None.
        # Disallow strings, bytes, ellipses from being primary constants.
        # math.nan and math.inf are floats, so they are allowed.
        if not isinstance(node.value, (int, float, bool)) and node.value is not None:
            self.safe = False
            return
        # No call to self.generic_visit(node) for Constant as it's a leaf.

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            # Disallow calls on attributes (obj.method()) or other complex calls
            self.safe = False
            return
        
        if node.func.id not in SAFE_GLOBALS:
            self.safe = False
            return
        
        # Check for *args and **kwargs, which are disallowed for simplicity
        # For Python 3.5+, *args would be an ast.Starred node in node.args.
        # ast.Starred is not in ALLOWED_NODE_TYPES, so it's covered.
        # For **kwargs (dict unpacking in call, e.g. foo(**{'bar':1})):
        for keyword_node in node.keywords:
            if keyword_node.arg is None: # This means **some_dict
                self.safe = False
                return
        
        # Recursively visit arguments and keyword arguments
        self.generic_visit(node)


    def visit_Attribute(self, node):
        # Disallow all attribute access (e.g., obj.attr)
        self.safe = False
        return


def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression from a string.

    Args:
        math_input: A string containing a Python mathematical expression.

    Returns:
        The result of the safe evaluation as a string, or
        'Unsafe Code Detected!' if any dangerous content is found
        or if the expression fails during evaluation.
    """
    if not isinstance(math_input, str):
        return "Unsafe Code Detected!"

    try:
        # Parse the input string into an AST. 'eval' mode for expressions.
        tree = ast.parse(math_input, mode='eval')
    except SyntaxError:
        return "Unsafe Code Detected!"

    visitor = SafeExpressionVisitor()
    try:
        visitor.visit(tree)
    except Exception: 
        # Catch any unexpected errors during AST traversal
        return "Unsafe Code Detected!"

    if not visitor.safe:
        return "Unsafe Code Detected!"

    # If the AST is deemed safe, try to evaluate it
    try:
        # Compile the AST to a code object before evaluating
        # filename='<user_expr>' is a convention for dynamically generated code
        code = compile(tree, filename='<user_expr>', mode='eval')
        
        # Evaluate the expression with the restricted globals and no locals
        result = eval(code, SAFE_GLOBALS_FOR_EVAL, {})
        return str(result)
    except (NameError, TypeError, ZeroDivisionError, OverflowError, ValueError) as e:
        # Catch typical math errors or errors from misusing allowed functions.
        # This also handles cases like calling a non-callable (e.g. pi()).
        return "Unsafe Code Detected!"
    except Exception:
        # Catch any other unexpected errors during evaluation
        return "Unsafe Code Detected!"

if __name__ == '__main__':
    # Example Usage and Test Cases
    test_expressions = [
        "1 + 2",                            # Expected: 3
        "sqrt(16)",                         # Expected: 4.0
        "pi * 2",                           # Expected: 6.283185307179586
        "factorial(5)",                     # Expected: 120
        "log10(100)",                       # Expected: 2.0
        "sum([1, 2, 3, 4, 5])",             # Expected: 15
        "max(1, 10, 2)",                    # Expected: 10
        "1 / 0",                            # Expected: Unsafe Code Detected! (ZeroDivisionError)
        "1 + undefined_var",                # Expected: Unsafe Code Detected! (NameError in AST check)
        "__import__('os').system('clear')", # Expected: Unsafe Code Detected! (AST check)
        "open('file.txt', 'w')",            # Expected: Unsafe Code Detected! (AST check)
        "lambda x: x + 1",                  # Expected: Unsafe Code Detected! (AST check)
        "eval('1+1')",                      # Expected: Unsafe Code Detected! (AST check)
        "pow(2, 1000)",                     # Expected: Large number or inf
        "1 if True else 0",                 # Expected: 1
        "'hello' + 'world'",                # Expected: Unsafe Code Detected! (Constant type check)
        "sqrt(-1)",                         # Expected: Unsafe Code Detected! (ValueError from math.sqrt)
        "factorial(1.5)",                   # Expected: Unsafe Code Detected! (TypeError or ValueError from math.factorial)
        "a = 10",                           # Expected: Unsafe Code Detected! (SyntaxError)
        "some_object.attribute",            # Expected: Unsafe Code Detected! (AST check for Attribute)
        "getattr(math, 'sqrt')",            # Expected: Unsafe Code Detected! (AST check for Name 'getattr')
        "math.sqrt(4)",                     # Expected: Unsafe Code Detected! (AST check for Attribute 'math.sqrt')
        "True",                             # Expected: True
        "1 < 2 and 3 > 1",                  # Expected: True
        "1 < 2 < 3",                        # Expected: True (chained comparison)
        "abs(-5.5)",                        # Expected: 5.5
        "round(3.14159, 2)",                # Expected: 3.14
        "([1,2,3])[0]",                     # Expected: Unsafe Code Detected! (Subscript)
    ]

    for expr in test_expressions:
        result = math_expression_safety_check(expr)
        print(f"Expression: {expr:<40} Result: {result}")

    # Test for non-string input
    print(f"Expression: {{'a': 1}}:<40 Result: {math_expression_safety_check({'a': 1})}")
    # Test for extremely long string (basic check, actual limits depend on system)
    # try:
    #     long_expr = "1+" * 100000 + "1" # Approx 200KB string
    #     # print(f"Expression: <long_expr>:<40 Result: {math_expression_safety_check(long_expr)}") # This can be slow
    # except MemoryError:
    #     print("Expression: <long_expr>:<40 Result: MemoryError during string creation")

    # Test for recursion depth in AST
    # deep_expr = "sqrt(" * 1000 + "4" + ")" * 1000 # Default recursion limit for AST is ~90-100 for parse
    # print(f"Expression: <deep_expr>:<40 Result: {math_expression_safety_check(deep_expr)}")
    # Python's parser has a fixed limit for parsing deeply nested expressions.
    # ast.parse("("*100 + "1" + ")"*100, mode='eval') -> s_push: parser stack overflow
    # This will result in SyntaxError, caught as "Unsafe Code Detected!"

    # Test for call to non-callable in SAFE_GLOBALS
    print(f"Expression: {'pi()':<40} Result: {math_expression_safety_check('pi()')}") # Expected: Unsafe Code Detected! (TypeError)
    
    # Test for disallowed constant type
    string_expr_result = math_expression_safety_check("'a string'")
    print(f"Expression: \"'a string'\" {' ':<27} Result: {string_expr_result}") # Expected: Unsafe Code Detected!
