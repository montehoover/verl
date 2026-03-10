import ast

# Allowed AST node classes. This whitelist determines what kind of language constructs are permitted.
_ALLOWED_NODE_CLASSES = {
    ast.Expression,  # Represents the overall expression.
    ast.Constant,    # For literal values like numbers, True, False, None (Python 3.8+).
    ast.Num,         # For numbers (deprecated in Python 3.8, but included for compatibility).
    ast.Name,        # For names like 'True', 'False', 'None'.
    ast.Load,        # Context for loading a name (occurs with ast.Name).
    ast.BinOp,       # For binary operations (e.g., +, -, *, /, and, or).
    ast.UnaryOp,     # For unary operations (e.g., -, not).
    ast.Compare,     # For comparison operations (e.g., ==, <, >).
    ast.BoolOp,      # For boolean operations like 'and', 'or'.
    ast.IfExp        # For ternary conditional expressions (e.g., x if C else y).
}

# Allowed operators for ast.BinOp nodes.
_ALLOWED_BIN_OPERATORS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv
}

# Allowed operators for ast.UnaryOp nodes.
_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd, ast.USub, ast.Not
}

# Allowed operators for ast.BoolOp nodes (e.g., 'and', 'or').
_ALLOWED_BOOL_OPERATORS = {
    ast.And, ast.Or
}

# Allowed operators for ast.Compare nodes.
_ALLOWED_COMPARE_OPERATORS = {
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE
}

# Safe local environment for eval, providing access to True, False, None.
_SAFE_LOCALS = {
    'True': True,
    'False': False,
    'None': None
}

def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluates a user-supplied mathematical or logical expression string.

    Args:
        math_expr: The string expression to evaluate.

    Returns:
        The string representation of the result, or 'Processing Failed!'
        if the expression is invalid, unsafe, or causes an error.
    """
    try:
        if not isinstance(math_expr, str) or not math_expr.strip():
            return 'Processing Failed!'

        # Parse the expression into an Abstract Syntax Tree (AST).
        # mode='eval' ensures it's an expression, not statements.
        tree = ast.parse(math_expr, mode='eval')

        # Validate all nodes in the AST.
        for node in ast.walk(tree):
            node_type = type(node)

            if node_type not in _ALLOWED_NODE_CLASSES:
                return 'Processing Failed!' # Disallowed node type

            # Specific checks for node types
            if node_type is ast.Constant:
                # Ensure constant values are of allowed types (numbers, booleans, None).
                # Disallows string literals, byte literals, etc.
                if not isinstance(node.value, (int, float, bool, type(None))):
                    return 'Processing Failed!'
            elif node_type is ast.Num: # For Python < 3.8
                if not isinstance(node.n, (int, float)):
                    return 'Processing Failed!'
            elif node_type is ast.Name:
                # Only allow 'True', 'False', 'None' as names.
                if node.id not in _SAFE_LOCALS:
                    return 'Processing Failed!'
            elif node_type is ast.BinOp:
                if type(node.op) not in _ALLOWED_BIN_OPERATORS:
                    return 'Processing Failed!'
            elif node_type is ast.UnaryOp:
                if type(node.op) not in _ALLOWED_UNARY_OPERATORS:
                    return 'Processing Failed!'
            elif node_type is ast.BoolOp:
                if type(node.op) not in _ALLOWED_BOOL_OPERATORS:
                    return 'Processing Failed!'
            elif node_type is ast.Compare:
                for op_type in node.ops: # node.ops is a list of comparison operators
                    if type(op_type) not in _ALLOWED_COMPARE_OPERATORS:
                        return 'Processing Failed!'
            # ast.Expression, ast.Load, ast.IfExp do not need further specific checks here
            # as their sub-nodes are also walked and validated by ast.walk.

        # If all nodes are valid, compile the AST to a code object.
        # '<user_expression>' is a dummy filename for error messages.
        code = compile(tree, filename='<user_expression>', mode='eval')

        # Evaluate the compiled code in a restricted environment.
        # __builtins__ is empty to prevent access to built-in functions.
        # _SAFE_LOCALS provides 'True', 'False', 'None'.
        result = eval(code, {'__builtins__': {}}, _SAFE_LOCALS)
        
        return str(result)

    except (SyntaxError, TypeError, NameError, ValueError, OverflowError,
            ZeroDivisionError, RecursionError, MemoryError):
        # Catch common errors during parsing or evaluation.
        return 'Processing Failed!'
    except Exception:
        # Catch any other unexpected errors as a security precaution.
        return 'Processing Failed!'

if __name__ == '__main__':
    # Example Usage:
    expressions = [
        "(10 + 5) - 3",              # Expected: 12
        "2 * (3 + 4) / 7",           # Expected: 2.0
        "2**3",                      # Expected: 8
        "10 > 5",                    # Expected: True
        "(2 > 1) and (3 < 5)",       # Expected: True
        "10 if (100 > 10) else 20",  # Expected: 10
        "1 / 0",                     # Expected: Processing Failed!
        "__import__('os').system('clear')", # Expected: Processing Failed!
        "a = 10; a + 5",             # Expected: Processing Failed! (SyntaxError for 'eval')
        "abs(-5)",                   # Expected: Processing Failed! (abs not allowed)
        "'hello' + 'world'",         # Expected: Processing Failed! (string literals not allowed)
        "bad_name",                  # Expected: Processing Failed! (Name node not in _SAFE_LOCALS)
        "",                          # Expected: Processing Failed!
        "   ",                       # Expected: Processing Failed!
        "10 + ",                     # Expected: Processing Failed! (SyntaxError)
    ]

    for expr_str in expressions:
        output = evaluate_user_math(expr_str)
        print(f"Input: \"{expr_str}\", Output: \"{output}\"")

    # Test with a potentially very long expression (careful with system limits)
    # try:
    #     long_expr = "1" + "+1" * 10000 # Creates a very long string "1+1+1+...+1"
    #     # This might be slow or hit recursion limits depending on Python's AST/parser limits
    #     # print(f"Input: \"long_expr (len: {len(long_expr)})\", Output: \"{evaluate_user_math(long_expr)}\"")
    # except RecursionError:
    #     print(f"Input: \"long_expr\", Output: \"Processing Failed! (RecursionError)\"")

    # Test with non-string input
    # print(f"Input: 123 (int), Output: \"{evaluate_user_math(123)}\"") # Expected: Processing Failed!
