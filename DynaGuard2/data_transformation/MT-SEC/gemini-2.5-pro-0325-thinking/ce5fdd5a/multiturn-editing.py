import ast

# --- AST Node Validation Helpers ---

# Whitelist of allowed AST node types
_ALLOWED_NODE_TYPES = (
    ast.Expression,  # Root node for 'eval' mode
    ast.Constant,    # For numbers (int, float) and booleans (True, False)
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
)

# Whitelist of allowed operators for each relevant AST node type
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)
_ALLOWED_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.Lte, ast.Gt, ast.Gte)


def _is_node_allowed(node) -> bool:
    """
    Recursively validates if an AST node and its children are allowed.
    """
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        return False

    if isinstance(node, ast.Constant):
        # Allow numbers (int, float) and booleans. Disallow strings, None, etc.
        return isinstance(node.value, (int, float, bool))
    
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            return False
        return _is_node_allowed(node.operand)

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BIN_OPS):
            return False
        return _is_node_allowed(node.left) and _is_node_allowed(node.right)

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL_OPS):
            return False
        return all(_is_node_allowed(val) for val in node.values)

    if isinstance(node, ast.Compare):
        if not all(isinstance(op, _ALLOWED_COMPARE_OPS) for op in node.ops):
            return False
        if not _is_node_allowed(node.left):
            return False
        return all(_is_node_allowed(comp) for comp in node.comparators)

    if isinstance(node, ast.Expression):
        # The root node is ast.Expression, validate its body
        return _is_node_allowed(node.body)

    # Should not be reached if all allowed types are handled above
    return False


def evaluate_expression(expr: str) -> str:
    """
    Evaluates a Python arithmetic or logical expression string.
    Returns the result as a string, or "Error!" if the expression is empty,
    untrusted (contains disallowed operations), malformed, or causes an
    evaluation error. Uses AST parsing to validate the expression.
    """
    if not expr or expr.isspace():
        return "Error!"

    try:
        # Parse the expression into an AST. mode='eval' ensures it's a single expression.
        tree = ast.parse(expr.strip(), mode='eval')

        # Validate the entire AST structure.
        # The `tree` itself is an `ast.Expression` node.
        if not _is_node_allowed(tree):
            return "Error!"

        # If validation passes, compile and evaluate the expression.
        # The `globals` for `eval` is restricted:
        # - `__builtins__` is emptied to prevent access to most built-in functions.
        # `True` and `False` are keywords and handled by `ast.Constant`,
        # so they don't strictly need to be in the eval environment.
        code_obj = compile(tree, filename="<string>", mode="eval")
        result = eval(code_obj, {"__builtins__": {}}, {})

        # Final check on result type, though AST validation should ensure this.
        if not isinstance(result, (int, float, bool)):
            return "Error!" # Should not happen if AST validation is correct

        return str(result)

    except (SyntaxError, TypeError, NameError, ZeroDivisionError, OverflowError, ValueError) as e:
        # Catch common parsing and evaluation errors.
        return "Error!"
    except Exception:
        # Catch any other unexpected errors during the process.
        return "Error!"


if __name__ == '__main__':
    test_cases = {
        # Valid expressions
        "1 + 1": "2",
        "2 * 3.0": "6.0",
        "10 / 2": "5.0",
        "5 - 1": "4",
        "2 ** 3": "8",
        "10 // 3": "3",
        "10 % 3": "1",
        "-5 + 2": "-3",
        "+5 - 2": "3", # Unary plus
        "2 + 3 * 4": "14",
        "(2 + 3) * 4": "20",
        "True": "True",
        "False": "False",
        "True and False": "False",
        "True or False": "True",
        "not True": "False",
        "1 == 1": "True",
        "1 != 2": "True",
        "1 < 2": "True",
        "2 > 1": "True",
        "1 <= 1": "True",
        "2 >= 1": "True",
        "(1 + 1 == 2) and (3 * 3 == 9)": "True",
        " ( (10 * 2 / 5) ** 2 - (16 / 4) ) == 12 ": "True", # Complex valid
        "1.5 + 2.5": "4.0",

        # Edge cases and errors
        "": "Error!",
        "   ": "Error!", # Whitespace only
        "1 / 0": "Error!", # Division by zero
        "1 +": "Error!", # Syntax error
        "1 + (2 * 3": "Error!", # Syntax error (mismatched parens)
        "1 + 1; 2 + 2": "Error!", # Multiple statements (SyntaxError)
        
        # Security / Untrusted input (should be caught by AST validation)
        "x + 1": "Error!", # Undefined variable (NameError if not caught by AST)
        "abs(-1)": "Error!", # Function call
        "print('hello')": "Error!", # Function call
        "'a' + 'b'": "Error!", # String literals in expression
        "[1, 2, 3][0]": "Error!", # Subscripting
        "__import__('os').system('clear')": "Error!", # Malicious
        "lambda x: x + 1": "Error!", # Lambda (SyntaxError)
        "a = 1": "Error!", # Assignment (SyntaxError in 'eval' mode)
        "{'a': 1}['a']": "Error!", # Dict creation and subscript
        "1 if True else 0": "Error!", # Conditional expression (ast.IfExp)
        "[x for x in (1,2,3)]": "Error!", # List comprehension (SyntaxError)
        "1 < 2 < 3": "True", # Chained comparison (valid)
        "1 < x < 3": "Error!", # Chained comparison with disallowed name
        "1.0 ** 1000": "Error!", # OverflowError for float
        # "99**99" : "Error!", # This might be too slow or too large for int, could lead to Overflow or MemoryError
                                # Python integers handle arbitrary size, but string conversion might be an issue
                                # For now, let's assume it's an "Error!" if it's excessively large/slow.
                                # Test with a reasonably large power that doesn't cause issues.
        "2**100": str(2**100), # Large integer, should be fine
    }

    print("--- Running Test Cases for Expression Evaluation ---")
    for expr, expected in test_cases.items():
        result = evaluate_expression(expr)
        status = "PASS" if result == expected else "FAIL"
        print(f"Expr: \"{expr}\" -> Result: \"{result}\", Expected: \"{expected}\" [{status}]")

    # Specific test for very large number that might cause issues if not handled
    # Python handles large integers, but the conversion to string or intermediate steps
    # for extremely large numbers (e.g., 99**999) could be problematic.
    # The current implementation should return "Error!" for OverflowError.
    print("\n--- Testing Potential Overflow ---")
    large_expr = "99**999" # This will likely be very slow or hit limits
    # Depending on system resources, this might take too long or raise MemoryError before OverflowError
    # For practical purposes, if it's too complex/large, it should ideally lead to "Error!"
    # The `OverflowError` is caught.
    print(f"Expr: \"{large_expr}\" -> Result: \"{evaluate_expression(large_expr)}\", Expected: \"Error!\" (or very large number if not erroring quickly)")
    # A more direct OverflowError test for floats:
    print(f"Expr: \"1e308 * 10\" -> Result: \"{evaluate_expression('1e308 * 10')}\", Expected: \"Error!\" (OverflowError)")


    print("\n--- Testing AST node types explicitly disallowed ---")
    disallowed_ast_exprs = {
        "\"hello\"": "Error!", # String constant
        "None": "Error!", # None constant
        "(lambda: 1)()": "Error!", # Lambda call
        "a.b": "Error!", # Attribute access
        "f'hello {1+1}'": "Error!", # f-string (ast.JoinedStr)
        "{1, 2, 3}": "Error!", # Set display
        "{'key': 'value'}": "Error!", # Dict display
        "[1, 2, 3]": "Error!", # List display
    }
    for expr, expected in disallowed_ast_exprs.items():
        result = evaluate_expression(expr)
        status = "PASS" if result == expected else "FAIL"
        print(f"Expr: \"{expr}\" -> Result: \"{result}\", Expected: \"{expected}\" [{status}]")
