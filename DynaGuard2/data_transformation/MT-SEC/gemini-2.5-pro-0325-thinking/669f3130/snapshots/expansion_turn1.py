import ast

# Base list of allowed AST node types that are common across Python versions
# and represent the structure of arithmetic expressions.
_ALLOWED_NODE_TYPES_LIST = [
    ast.Expression,  # The root node of an expression parsed with mode='eval'.
    ast.BinOp,       # For binary operations like +, -, *, /.
    ast.UnaryOp,     # For unary operations like - (negation) or + (unary plus).
    ast.Add,         # The addition operator type.
    ast.Sub,         # The subtraction operator type.
    ast.Mult,        # The multiplication operator type.
    ast.Div,         # The division operator type.
    ast.UAdd,        # The unary plus operator type.
    ast.USub,        # The unary minus operator type.
]

# Add AST node types for numeric literals, handling Python version differences.
# Python 3.8+ uses ast.Constant for numbers, strings, None, True, False.
# Python < 3.8 uses ast.Num for numbers and ast.Constant for None, True, False.
# ast.NameConstant was used for None, True, False in Python < 3.8, and ast.Constant was an alias in 3.6, 3.7.
# To be precise and cover common versions (e.g., 3.6 to latest):
if hasattr(ast, 'Constant'):
    # ast.Constant exists in Python 3.6+.
    # In 3.6/3.7, it's for None/True/False.
    # In 3.8+, it's for numbers, strings, bytes, None, True, False.
    _ALLOWED_NODE_TYPES_LIST.append(ast.Constant)
if hasattr(ast, 'Num'):
    # ast.Num exists in Python <= 3.7 for numbers.
    # In Python 3.8, ast.Num is an alias for ast.Constant.
    # ast.Num is removed in Python 3.9+.
    _ALLOWED_NODE_TYPES_LIST.append(ast.Num)

# Convert the list to a tuple for efficient use with isinstance().
_ALLOWED_NODE_TYPES_TUPLE = tuple(set(_ALLOWED_NODE_TYPES_LIST)) # Use set to remove duplicates if any (e.g. ast.Num alias in 3.8)


def is_safe_expression(expression: str) -> bool:
    """
    Checks if the given arithmetic expression string is safe to evaluate.

    A safe expression can only contain:
    - Numbers (integers and floating-point).
    - Parentheses for grouping.
    - The basic arithmetic operations: addition (+), subtraction (-),
      multiplication (*), and division (/).
    - Unary plus (+) and unary minus (-).

    The function parses the expression into an Abstract Syntax Tree (AST)
    and verifies that all nodes in the tree correspond to allowed elements.
    This helps prevent evaluation of potentially harmful code, such as
    function calls, variable names, or other complex Python constructs.

    Args:
        expression: The user-provided arithmetic expression string.

    Returns:
        True if the expression is deemed safe, False otherwise.
        False is also returned for invalid syntax or non-string inputs.
    """
    if not isinstance(expression, str):
        return False  # Input must be a string.
    
    if not expression.strip():
        return False  # Empty or whitespace-only strings are not valid expressions.

    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False  # The expression has invalid Python syntax.
    except ValueError: # Handles null bytes in expression
        return False

    for node in ast.walk(tree):
        # Check if the node type is in our whitelist of allowed types.
        if not isinstance(node, _ALLOWED_NODE_TYPES_TUPLE):
            return False  # Disallowed node type found.

        # Additional checks for specific node types:
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            # If ast.Constant is used (Python 3.6+), ensure its value is a number.
            # This prevents evaluation of strings, None, True, False if they were
            # to be represented by ast.Constant in some Python version/context.
            if not isinstance(node.value, (int, float)):
                return False  # ast.Constant holding non-numeric data.
        
        # No specific check needed for ast.Num, as it inherently represents a number
        # in Python versions where it's distinct from ast.Constant (i.e., < 3.8).
        # If ast.Num is an alias for ast.Constant (Python 3.8), the ast.Constant check above applies.

        # Note: The operator types (ast.Add, ast.Sub, etc.) are themselves nodes
        # and are included in _ALLOWED_NODE_TYPES_TUPLE.
        # ast.BinOp and ast.UnaryOp nodes have an 'op' attribute which is an instance
        # of these operator types. ast.walk visits these 'op' nodes too.

    return True  # All nodes are of allowed types and satisfy specific checks.

if __name__ == '__main__':
    # Example Usage and Test Cases
    safe_expressions = [
        "1 + 2",
        "10 - 3.5",
        "2 * (3 + 4)",
        "100 / 5",
        "-5",
        "+3.14",
        "((1 + 2) * 3 - (4 / 2))",
        "1.0 * 2.0 / 3.0 + 4.0 - 5.0",
        "42",
        "  1 + 1  ", # Whitespace around
    ]

    unsafe_expressions = [
        "__import__('os').system('clear')", # Malicious code
        "print('hello')",                   # Function call
        "x + 1",                            # Variable name
        "1 ** 2",                           # Power operator (not allowed)
        "1 % 2",                            # Modulo operator (not allowed)
        "eval('1+1')",                      # eval itself
        "a.b",                              # Attribute access
        "[1, 2, 3]",                        # List literal
        "{'a': 1}",                         # Dict literal
        "lambda x: x + 1",                  # Lambda function
        "1 if True else 0",                 # Conditional expression
        "'hello' + 'world'",                # String concatenation
        "True",                             # Boolean literal
        "None",                             # None literal
        "",                                 # Empty string
        "   ",                              # Whitespace only string
        "1 + ",                             # Incomplete expression (SyntaxError)
        "1 + * 2",                          # Invalid syntax (SyntaxError)
        "1 + (2 * 3",                       # Mismatched parentheses (SyntaxError)
        "import os",                        # Import statement (not 'eval' mode, but good to list)
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        print(f"'{expr}': {is_safe_expression(expr)}")

    # Test non-string input
    print("\nTesting non-string input:")
    print(f"123 (int): {is_safe_expression(123)}")
    print(f"None: {is_safe_expression(None)}")
