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
    # ALLOWED_NODE_TYPES: A set of AST node types that are considered safe for mathematical expressions.
    #                     This whitelist prevents execution of arbitrary Python code by ensuring that
    #                     only expected and safe AST nodes are present in the parsed expression.
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
        # Overrides the default visitor method.
        # If an unsafe node has already been detected (self.safe is False), stop visiting.
        # If the current node's type is not in ALLOWED_NODE_TYPES, mark the expression as
        # unsafe and stop further traversal of this branch.
        if not self.safe:  # If already found unsafe, stop visiting
            return

        if type(node) not in self.ALLOWED_NODE_TYPES:
            self.safe = False
            return

        super().visit(node)  # Continue visiting children if current node is safe

    def visit_Name(self, node):
        # Ensures that any variable name (e.g., 'pi') or function name (e.g., 'sqrt')
        # used in the expression is present in the SAFE_GLOBALS whitelist.
        # This prevents access to unsafe builtins (like 'eval' or 'open') or other
        # global variables not explicitly allowed.
        if node.id not in SAFE_GLOBALS:
            self.safe = False
            return
        self.generic_visit(node) # Ensure child nodes are also visited if any (e.g. in different AST contexts)

    def visit_Constant(self, node):
        # Allows only numeric (int, float), boolean (True, False), and None constants.
        # Strings, bytes, or other complex constant types are disallowed to prevent
        # potential injection vectors or unintended behavior (e.g., constructing code from strings).
        # math.nan and math.inf are floats, so they are permitted by isinstance(node.value, float).
        if not isinstance(node.value, (int, float, bool)) and node.value is not None:
            self.safe = False
            return
        # No call to self.generic_visit(node) for Constant as it's a leaf node in the AST.

    def visit_Call(self, node):
        # Validates function calls (e.g., `sqrt(4)`).
        # 1. Ensures that the function being called is a direct name (ast.Name),
        #    not an attribute (e.g., `obj.method()`) or other complex expression.
        # 2. Checks if the function name (node.func.id) is in SAFE_GLOBALS.
        # 3. Disallows argument unpacking (*args, **kwargs) for simplicity and to avoid
        #    potential attack vectors if ast.Starred or keyword.arg is None were mishandled.
        #    ast.Starred nodes for *args are caught by the main `visit` method as they
        #    are not in ALLOWED_NODE_TYPES.
        if not isinstance(node.func, ast.Name):
            # Disallow calls on attributes (obj.method()) or other complex call targets
            self.safe = False
            return
        
        if node.func.id not in SAFE_GLOBALS:
            self.safe = False
            return
        
        # Check for **kwargs (dict unpacking in call, e.g. foo(**{'bar':1})).
        # node.keywords is a list of ast.keyword.
        # If keyword.arg is None, it means **some_dict, which is disallowed.
        for keyword_node in node.keywords:
            if keyword_node.arg is None: # This means **some_dict
                self.safe = False
                return
        
        # Recursively visit arguments and keyword arguments' values
        self.generic_visit(node)


    def visit_Attribute(self, node):
        # Disallows all attribute access (e.g., `object.attribute` or `math.pi`).
        # This is a critical security measure to prevent access to methods or properties
        # that could lead to unsafe operations (e.g., `"".__class__`, `().__doc__`, etc.).
        # All allowed functions and constants must be accessed directly by name from SAFE_GLOBALS.
        self.safe = False
        return


# Helper functions for parsing, validation, and evaluation

def _parse_expression_to_ast(math_input: str) -> ast.AST | None:
    """
    Parses a mathematical expression string into an Abstract Syntax Tree (AST).
    Catches SyntaxError if parsing fails or other exceptions for robustness.

    Args:
        math_input: The string containing the mathematical expression.

    Returns:
        The AST object if parsing is successful, None otherwise.
    """
    try:
        # Parse the input string into an AST. 'eval' mode is used because we expect
        # an expression that results in a value, not a statement.
        tree = ast.parse(math_input, mode='eval')
        return tree
    except SyntaxError: # Common case for invalid Python syntax
        return None
    except Exception: # Catch other potential parsing issues (e.g., recursion depth limits in parser)
        return None

def _validate_ast(tree: ast.AST) -> bool:
    """
    Validates an AST using SafeExpressionVisitor to ensure it contains only allowed constructs.
    Catches any exceptions during AST traversal.

    Args:
        tree: The AST object to validate.

    Returns:
        True if the AST is safe (contains only whitelisted nodes and names), False otherwise.
    """
    visitor = SafeExpressionVisitor()
    try:
        visitor.visit(tree)
        return visitor.safe
    except Exception: 
        # Catch any unexpected errors during AST traversal (e.g., recursion depth in visitor)
        return False

def _evaluate_safe_ast(tree: ast.AST) -> object | None:
    """
    Compiles and evaluates a validated AST in a restricted environment.
    Catches common evaluation errors (like ZeroDivisionError) and other exceptions.

    Args:
        tree: The validated AST object.

    Returns:
        The result of the evaluation if successful, None otherwise. The type of the
        result can be any Python object (e.g., int, float, bool).
    """
    try:
        # Compile the AST to a code object. This step can also catch some errors.
        # filename='<user_expr>' is a convention for dynamically generated code.
        code = compile(tree, filename='<user_expr>', mode='eval')
        
        # Evaluate the compiled code.
        # SAFE_GLOBALS_FOR_EVAL provides the execution context (allowed functions/constants).
        # An empty dictionary {} is passed for locals, further restricting the context.
        result = eval(code, SAFE_GLOBALS_FOR_EVAL, {})
        return result
    except (NameError, TypeError, ZeroDivisionError, OverflowError, ValueError) as e:
        # Catch typical math errors or errors from misusing allowed functions
        # (e.g., calling a non-callable like pi(), or sqrt of a negative number).
        return None
    except Exception:
        # Catch any other unexpected errors during compilation or evaluation.
        return None

def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression from a string by parsing,
    validating, and then evaluating it in a controlled environment.

    The process involves three main steps:
    1. Parsing: The input string is parsed into an Abstract Syntax Tree (AST).
    2. Validation: The AST is traversed to ensure it only contains allowed operations,
       functions, and constants, preventing the use of unsafe constructs.
    3. Evaluation: If the AST is valid, the expression is compiled and evaluated
       within a restricted scope that only exposes safe mathematical functions.

    Args:
        math_input: A string containing a Python mathematical expression.

    Returns:
        The result of the safe evaluation as a string, or
        'Unsafe Code Detected!' if any dangerous content is found,
        or if the expression fails during parsing, validation, or evaluation.
    """
    # Initial type check for the input: must be a string.
    if not isinstance(math_input, str):
        return "Unsafe Code Detected!"

    # Step 1: Parse the expression string to an AST.
    # If parsing fails (e.g., due to syntax errors or overly complex structures),
    # _parse_expression_to_ast returns None.
    ast_tree = _parse_expression_to_ast(math_input)
    if ast_tree is None:
        return "Unsafe Code Detected!"

    # Step 2: Validate the AST for allowed constructs.
    # _validate_ast returns False if the AST contains any disallowed elements
    # (like unauthorized function calls, attribute access, or unsafe node types).
    if not _validate_ast(ast_tree):
        return "Unsafe Code Detected!"

    # Step 3: Evaluate the validated AST.
    # _evaluate_safe_ast returns None if evaluation fails (e.g., due to mathematical
    # errors like division by zero, or other runtime issues within the safe functions).
    result = _evaluate_safe_ast(ast_tree)
    if result is None:
        return "Unsafe Code Detected!"

    # If all steps pass, convert the result to a string and return it.
    return str(result)

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
