import ast
import re

# Pattern for allowed characters: numbers, operators (+, -, *, /), parentheses, decimal point, and whitespace.
# Using re.compile for slight performance benefit if called many times, though not strictly necessary here.
_ALLOWED_CHAR_PATTERN = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

# Allowed AST node types at the expression level (e.g., parts of an expression)
_ALLOWED_EXPR_NODE_TYPES = (
    ast.Constant,  # For numeric literals (integers, floats)
    ast.BinOp,     # For binary operations (e.g., a + b)
    ast.UnaryOp,   # For unary operations (e.g., -a)
)

# Allowed operator types for binary operations (ast.BinOp.op)
_ALLOWED_BIN_OP_TYPES = (
    ast.Add,   # +
    ast.Sub,   # -
    ast.Mult,  # *
    ast.Div,   # /
)

# Allowed operator types for unary operations (ast.UnaryOp.op)
_ALLOWED_UNARY_OP_TYPES = (
    ast.UAdd,  # Unary +
    ast.USub,  # Unary -
)


def _is_safe_ast_node(node: ast.AST) -> bool:
    """
    Recursively validates an AST node and its children.
    Ensures that the node type and its properties (like operators or constant values)
    are within the defined safe list for basic arithmetic.
    """
    if isinstance(node, ast.Expression):
        # For an ast.Expression node (root from 'eval' mode), validate its body.
        return _is_safe_ast_node(node.body)

    if not isinstance(node, _ALLOWED_EXPR_NODE_TYPES):
        # Node type is not in the allowed list (e.g., ast.Call, ast.Name).
        return False

    if isinstance(node, ast.Constant):
        # For constants, ensure the value is a number (integer or float).
        return isinstance(node.value, (int, float))

    elif isinstance(node, ast.BinOp):
        # For binary operations, check the operator type and recursively validate operands.
        if not isinstance(node.op, _ALLOWED_BIN_OP_TYPES):
            return False
        return _is_safe_ast_node(node.left) and _is_safe_ast_node(node.right)

    elif isinstance(node, ast.UnaryOp):
        # For unary operations, check the operator type and recursively validate the operand.
        if not isinstance(node.op, _ALLOWED_UNARY_OP_TYPES):
            return False
        return _is_safe_ast_node(node.operand)

    # Fallback, though theoretically unreachable if _ALLOWED_EXPR_NODE_TYPES is comprehensive
    # for the types we encounter after initial checks.
    return False


def validate_expression(expression: str) -> bool:
    """
    Validates a user-submitted mathematical expression string.

    The function checks if the expression:
    1. Is a string.
    2. Contains only allowed characters (digits, whitespace, '.', '+', '-', '*', '/', '(', ')').
    3. Is not empty or only whitespace.
    4. Is syntactically valid Python for a single expression.
    5. Consists only of AST nodes corresponding to basic arithmetic operations
       (numbers, +, -, *, /, unary +, unary -) and parentheses.
       It explicitly disallows variables, function calls, and other potentially unsafe constructs.

    Args:
        expression: The mathematical expression string to validate.

    Returns:
        True if the expression is valid and safe for evaluation using basic arithmetic rules,
        False otherwise.
    """
    if not isinstance(expression, str):
        return False

    # 1. Preliminary check for allowed characters.
    #    This is a quick filter for obviously invalid or malicious strings.
    if not _ALLOWED_CHAR_PATTERN.fullmatch(expression):
        return False

    # 2. Handle empty or whitespace-only strings.
    #    An empty string or one with only whitespace is not a valid arithmetic expression.
    stripped_expression = expression.strip()
    if not stripped_expression:
        return False

    # 3. Attempt to parse the expression into an Abstract Syntax Tree (AST).
    #    'eval' mode is used because we expect a single expression.
    #    If parsing fails, it's not a valid Python expression.
    try:
        tree = ast.parse(stripped_expression, mode='eval')
    except SyntaxError:
        # Catches errors like "1 +", "1.2.3", unbalanced parentheses, etc.
        return False
    except Exception:
        # Catches other potential parsing issues (e.g., recursion depth limits for overly complex inputs).
        # Treat these as unsafe/invalid.
        return False

    # 4. Traverse the AST to ensure all nodes are safe and conform to basic arithmetic.
    #    The root of the tree from ast.parse(..., mode='eval') is an ast.Expression node.
    if not _is_safe_ast_node(tree):
        return False

    return True

if __name__ == '__main__':
    # Example Usage and Test Cases
    test_expressions = {
        "1 + 2": True,
        "   (3 * 4) - 5 / 2.0  ": True,
        "-5 + +3": True,
        "0.5 * (2 + 3)": True,
        "10 / 2": True,
        "100": True,
        "-3.14": True,
        "(1)": True,
        "1 + ": False,  # Syntax error
        "1 + 2a": False, # Disallowed character 'a'
        "import os": False, # Disallowed characters
        "__import__('os').system('clear')": False, # Disallowed characters
        "print('hello')": False, # Disallowed characters
        "abs(-5)": False, # Disallowed characters (implies function call)
        "10**2": False, # Disallowed operator '**' (ast.Pow not in allow list)
        "1 / 0": True,  # Syntactically valid; runtime ZeroDivisionError is separate
        "1.2.3 + 4": False, # Syntax error
        "": False, # Empty string
        "   ": False, # Whitespace only
        "foo + bar": False, # Disallowed characters (implies variables)
        "x = 1": False, # Disallowed character '=' (implies assignment)
        "1; 2": False, # Disallowed character ';' (implies multiple statements)
        "1 if True else 0": False, # Disallowed AST node type (ast.IfExp)
        "[1, 2, 3]": False, # Disallowed AST node type (ast.List)
        "1_000_000 + 1": True, # Numeric literals with underscores are fine (parsed as int)
        "1e5 + 1": True, # Scientific notation is fine (parsed as float)
    }

    all_passed = True
    for expr, expected in test_expressions.items():
        result = validate_expression(expr)
        if result == expected:
            print(f"PASS: validate_expression(\"{expr}\") == {expected}")
        else:
            print(f"FAIL: validate_expression(\"{expr}\") == {result} (expected {expected})")
            all_passed = False

    if all_passed:
        print("\nAll example tests passed!")
    else:
        print("\nSome example tests failed.")
