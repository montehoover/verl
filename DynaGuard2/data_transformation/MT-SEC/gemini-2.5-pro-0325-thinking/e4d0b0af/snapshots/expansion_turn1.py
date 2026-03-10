import ast

def is_expression_safe(expression: str) -> bool:
    """
    Checks if a string expression is safe by parsing it into an Abstract Syntax Tree (AST)
    and verifying that all nodes in the tree correspond to allowed, simple operations.

    Args:
        expression: The string input representing a mathematical or logical expression.

    Returns:
        True if the expression only contains safe arithmetic or logical operations
        on numeric or boolean literals. False otherwise, including if the expression
        is not valid Python syntax, or uses disallowed constructs like variables,
        function calls, attribute access, string literals, etc.

    Allowed constructs:
    - Numeric literals (integers, floats)
    - Boolean literals (True, False)
    - Arithmetic operators: +, -, *, /, // (floor division), % (modulo), ** (power)
    - Unary operators: - (negation), + (unary plus), not (logical not)
    - Comparison operators: ==, !=, <, <=, >, >=
    - Logical operators: and, or
    - Parentheses for grouping are implicitly handled by the AST structure.
    """
    try:
        # Parse the expression. mode='eval' ensures it's an expression, not statements.
        # If parsing fails, it's not a valid Python expression.
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False

    # Whitelist of allowed AST node types.
    # Operator nodes (e.g., ast.Add) are included as they appear as distinct nodes
    # in the AST walk, typically as attributes of operational nodes (e.g., ast.BinOp).
    allowed_node_types = (
        ast.Expression,  # The root node of an expression parsed with mode='eval'.

        # Literals
        ast.Constant,    # Represents literal values like numbers, booleans.
                         # String literals will be filtered out by a specific check.

        # Binary Operations (e.g., a + b)
        ast.BinOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,

        # Unary Operations (e.g., -a, not a)
        ast.UnaryOp,
        ast.UAdd, ast.USub, ast.Not,

        # Comparisons (e.g., a == b)
        ast.Compare,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,

        # Boolean Logic (e.g., a and b)
        ast.BoolOp,
        ast.And, ast.Or,
    )

    # Disallowed constructs (explicitly for clarity, though covered by whitelist):
    # - ast.Name (variables)
    # - ast.Call (function calls)
    # - ast.Attribute (attribute access, e.g., obj.method)
    # - ast.Subscript (indexing or slicing, e.g., my_list[0])
    # - ast.Lambda (lambda functions)
    # - ast.IfExp (ternary operator, e.g., x if condition else y)
    # - Comprehensions (ListComp, SetComp, DictComp, GeneratorExp)
    # - Statement types (Import, FunctionDef, ClassDef, Assign, etc. - already prevented by mode='eval')

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            # If node type is not in the whitelist, it's disallowed.
            return False

        # Additional specific checks for node types that require them:
        if isinstance(node, ast.Constant):
            # For ast.Constant, ensure the value is a number (int, float) or boolean.
            # This explicitly disallows string literals, None, bytes, Ellipsis from being constants.
            if not isinstance(node.value, (int, float, bool)):
                return False
        
        # No ast.Name nodes are allowed (i.e., no variables).
        # If ast.Name were in allowed_node_types, specific checks for node.id would be needed here.

    # If all nodes are of allowed types and satisfy specific checks, the expression is safe.
    return True

if __name__ == '__main__':
    # Example Usage and Test Cases
    safe_expressions = [
        "1 + 2",
        "10 * (2 - 3.5) / 1.0",
        "-5 + +3",
        "2 ** 3",
        "10 % 3",
        "10 // 3",
        "True and False",
        "not True",
        "1 < 2",
        "1 <= 2 and 3 > 2",
        "(1 == 1) or (2 != 3)",
        "1.0e-5", # Scientific notation for floats
    ]

    unsafe_expressions = [
        "x + 1",  # Variable
        "abs(-1)",  # Function call
        "__import__('os').system('clear')",  # Malicious call
        "eval('1+1')", # eval itself
        "my_object.attribute",  # Attribute access
        "my_list[0]",  # Subscripting
        "lambda x: x + 1",  # Lambda
        "'hello' + 'world'",  # String literal and concatenation
        "1 if True else 0", # If expression (ternary operator)
        "[1, 2, 3]", # List literal
        "{'a': 1}", # Dict literal
        "1; print('unsafe')", # Multiple statements (SyntaxError due to mode='eval')
        "", # Empty string (SyntaxError)
        "   ", # Whitespace only (SyntaxError)
        "1 + # comment", # Comment (SyntaxError in mode='eval' if not on its own line)
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        print(f"'{expr}': {is_expression_safe(expr)}")

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        print(f"'{expr}': {is_expression_safe(expr)}")

    # Test specific edge cases
    print("\nTesting specific edge cases:")
    # Python 3.8+ parses True/False as Constant(value=True/False)
    # Older versions might parse them as Name(id='True'/ 'False').
    # The current implementation relies on Constant for booleans.
    print(f"'True': {is_expression_safe('True')}") # Should be True
    print(f"'False': {is_expression_safe('False')}") # Should be True
    print(f"'None': {is_expression_safe('None')}") # Should be False (Constant(value=NoneType))
    print(f"'\"text\"': {is_expression_safe('\"text\"')}") # Should be False (Constant(value=str))

    # Example of an expression that is syntactically safe but would cause a runtime error
    # This function checks for structural safety, not runtime validity.
    runtime_error_expr = "1 / 0"
    print(f"'{runtime_error_expr}' (runtime error potential): {is_expression_safe(runtime_error_expr)}") # Should be True
