import ast
import sys

# Define the set of allowed AST node types for validating expressions.
# This set aims to allow common arithmetic and logical operations while
# disallowing potentially harmful constructs like function calls, imports,
# attribute access on arbitrary objects, etc.

ALLOWED_NODE_TYPES = {
    # Core structure
    ast.Expression,  # Represents the overall expression body

    # Literals
    # Modern Python (3.8+) uses ast.Constant for all constants (numbers, strings, bools, None).
    ast.Constant,
}

# For older Python versions (< 3.8), specific literal types were used.
# We add them if they exist in the current `ast` module to ensure compatibility.
if hasattr(ast, 'Num'): # Handles int, float, complex in Python < 3.8
    ALLOWED_NODE_TYPES.add(ast.Num)
if hasattr(ast, 'Str'): # Handles string literals in Python < 3.8
    ALLOWED_NODE_TYPES.add(ast.Str) 
    # Allowing strings for expressions like 'a' == 'b' or 'hello' + 'world'.
    # If only numeric/boolean literals are desired, ast.Str could be omitted,
    # and for ast.Constant, one might need to check node.value's type.
# ast.Bytes is intentionally omitted as byte literals are less common in typical
# user-facing arithmetic/logical expressions.
if hasattr(ast, 'NameConstant'): # Handles True, False, None in Python < 3.8
    ALLOWED_NODE_TYPES.add(ast.NameConstant)

# Variables and names
ALLOWED_NODE_TYPES.add(ast.Name)  # For variable names like x, y, var_name

# Operators and operation types
ALLOWED_NODE_TYPES.update({
    ast.BinOp,      # e.g., a + b, x * y
    ast.UnaryOp,    # e.g., -x, not y
    ast.BoolOp,     # e.g., x and y, a or b
    ast.Compare,    # e.g., x > y, a == b

    # Specific operator types (these are actual classes for ops, children of the above)
    # Binary operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,

    # Unary operators
    ast.UAdd, ast.USub, ast.Not, ast.Invert, # ast.Invert is bitwise NOT (~)

    # Boolean operators
    ast.And, ast.Or,

    # Comparison operators
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
})


def parse_expression(expression_string: str) -> bool:
    """
    Parses a user-supplied expression string and returns a boolean
    indicating whether the expression contains only valid arithmetic or
    logical operations.

    The validation is done by parsing the expression into an Abstract Syntax Tree (AST)
    and checking if all nodes in the tree belong to a predefined set of allowed
    node types. This helps prevent the execution of harmful commands or unintended
    operations.

    Args:
        expression_string: The string containing the expression to parse.

    Returns:
        True if the expression is valid and contains only allowed operations,
        False otherwise.
    """
    if not isinstance(expression_string, str):
        return False
    
    if not expression_string.strip():
        # Empty or whitespace-only string is not a valid expression.
        return False

    try:
        # Parse the expression string into an AST.
        # 'eval' mode is used because we expect a single expression, not statements.
        tree = ast.parse(expression_string, mode='eval')
    except SyntaxError:
        # If parsing fails due to syntax errors, it's not a valid expression.
        return False
    except Exception:
        # Catch other potential errors during parsing, e.g., recursion depth
        # for overly complex expressions, or memory errors for huge ones.
        return False

    # Traverse all nodes in the AST.
    for node in ast.walk(tree):
        # Check if the type of the current node is in our whitelist.
        if type(node) not in ALLOWED_NODE_TYPES:
            # If any node type is not allowed, the expression is considered invalid/unsafe.
            # For debugging purposes, one might log the disallowed node:
            # print(f"Disallowed node type: {type(node)} in expression: '{expression_string}'")
            return False
            
    # If all nodes are of allowed types, the expression is considered valid.
    return True
