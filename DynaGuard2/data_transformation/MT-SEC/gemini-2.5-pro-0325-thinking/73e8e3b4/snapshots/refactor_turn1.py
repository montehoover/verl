import re
import ast

def evaluate_and_compute(calc_expr: str):
    """
    Evaluates a mathematical expression and returns the result.

    Args:
        calc_expr: str, a string containing a mathematical expression to evaluate.

    Returns:
        The evaluated result of the expression.

    Raises:
        ValueError: If the expression includes unsupported operators, characters,
                    unsafe code, or if evaluation fails.
    """
    # Allow numbers, basic arithmetic operators, parentheses, and whitespace.
    # Disallow any letters or other symbols to prevent function calls or variable names.
    if not re.fullmatch(r"[0-9\s\.\+\-\*\/\(\)]+", calc_expr):
        raise ValueError("Expression contains unsupported characters.")

    try:
        # Parse the expression into an AST node
        node = ast.parse(calc_expr, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except Exception as e:
        raise ValueError(f"Error parsing expression: {e}")

    # Whitelist of allowed AST node types
    allowed_nodes = {
        ast.Expression,  # The top-level node for an expression
        ast.Num,         # Numbers (deprecated in Python 3.8, use ast.Constant)
        ast.Constant,    # Numbers, strings, None, True, False
        ast.BinOp,       # Binary operations like +, -, *, /
        ast.UnaryOp,     # Unary operations like - (negation)
        ast.Add,         # Addition operator
        ast.Sub,         # Subtraction operator
        ast.Mult,        # Multiplication operator
        ast.Div,         # Division operator
        ast.USub,        # Unary subtraction (negation)
        ast.UAdd,        # Unary addition (identity)
        # ast.Name could be allowed if we had a context of safe variables/constants
        # For now, disallow ast.Name to prevent arbitrary variable access.
        # ast.Call is disallowed to prevent function calls.
    }

    # Validate all nodes in the AST
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, tuple(allowed_nodes)):
            # Special handling for ast.Num in older Python versions if ast.Constant is preferred
            if isinstance(sub_node, ast.Num) and ast.Constant in allowed_nodes:
                continue # Allow ast.Num if ast.Constant is in allowed_nodes for compatibility
            raise ValueError(f"Unsupported operation or node type: {type(sub_node).__name__}")
        
        # Ensure ast.Constant is a number (int or float)
        if isinstance(sub_node, ast.Constant) and not isinstance(sub_node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")


    try:
        # Compile the AST node into a code object
        # The mode 'eval' is used because we expect a single expression
        code = compile(node, filename='<string>', mode='eval')
        
        # Evaluate the compiled code object
        # Provide an empty dictionary for globals and locals to restrict context
        result = eval(code, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Catch any other errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 1": 2,
        "10 - 5": 5,
        "3 * 7": 21,
        "20 / 4": 5.0,
        "(1 + 2) * 3": 9,
        "10 / (2 + 3)": 2.0,
        "-5 + 10": 5,
        "-(5 + 5)": -10,
        "3.14 * 2": 6.28,
        "100 / 0.5": 200.0,
    }

    print("Running tests...")
    for expr, expected in test_expressions.items():
        try:
            result = evaluate_and_compute(expr)
            print(f"Expression: '{expr}', Result: {result}, Expected: {expected} -> {'PASS' if result == expected else 'FAIL'}")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e} -> FAIL (unexpected error)")

    error_expressions = [
        "1 + a",  # unsupported character 'a'
        "import os", # unsupported operation
        "__import__('os').system('clear')", # unsafe code
        "1 / 0", # division by zero
        "1 +", # syntax error
        "eval('1+1')", # unsupported operation (Call)
        "print('hello')", # unsupported operation (Call)
        "abs(-5)", # unsupported operation (Call)
        "'hello' + 'world'", # non-numeric constant
    ]
    print("\nRunning error case tests...")
    for expr in error_expressions:
        try:
            evaluate_and_compute(expr)
            print(f"Expression: '{expr}' -> FAIL (expected ValueError)")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e} -> PASS (ValueError raised as expected)")
        except Exception as e:
            print(f"Expression: '{expr}', Unexpected Error: {e} -> FAIL")
