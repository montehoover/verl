import re
import ast

# Allowed characters: digits, operators (+, -, *, /), parentheses, and whitespace.
# This regex ensures that only these characters and basic structures are present.
# It's a basic sanitization step. More complex validation might be needed for production.
ALLOWED_PATTERN = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]*$")

# A list of AST node types that are allowed in the expression.
# This helps prevent execution of arbitrary code.
ALLOWED_NODES = {
    ast.Expression, ast.Num, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Call, ast.Name, ast.Load
}

# For ast.Call and ast.Name, we might want to restrict what functions/names are allowed.
# For this simple calculator, we won't allow any function calls or variable names.
# So, ast.Call and ast.Name will be disallowed by checking the node types directly.
# If we wanted to allow specific functions (e.g. 'abs'), we'd need more complex logic.

def _is_node_allowed(node):
    """Recursively check if all nodes in the AST are allowed."""
    if not isinstance(node, tuple(ALLOWED_NODES)):
        return False
    # Disallow function calls and variable names for this simple calculator
    if isinstance(node, (ast.Call, ast.Name)):
        return False
    for child_node in ast.iter_child_nodes(node):
        if not _is_node_allowed(child_node):
            return False
    return True


def parse_and_calculate(expression: str):
    """
    Processes a mathematical expression provided by a user,
    parsing the expression and performing the calculation.

    Args:
        expression: str, the arithmetic expression provided by the user.

    Returns:
        The result of the calculated expression.

    Raises:
        ValueError: If the input contains unsupported characters,
                    malicious commands, or cannot be computed.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    # 1. Validate allowed characters
    if not ALLOWED_PATTERN.fullmatch(expression):
        raise ValueError("Expression contains unsupported characters.")

    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # 2. Parse the expression into an AST
        # mode='eval' is used because we expect an expression that returns a value.
        node = ast.parse(expression, mode='eval')

        # 3. Validate AST nodes (security measure)
        if not _is_node_allowed(node):
            raise ValueError("Expression contains disallowed operations or constructs.")

        # 4. Compile the AST into a code object
        # <string> is a dummy filename for error messages.
        code = compile(node, filename='<expression>', mode='eval')

        # 5. Evaluate the compiled code object
        # We provide an empty dictionary for globals and locals to restrict the environment.
        # For a simple calculator, this is generally safe after AST validation.
        # More complex scenarios might require a custom evaluation visitor for the AST.
        result = eval(code, {"__builtins__": {}}, {})
        
        # Ensure result is a number (int or float)
        if not isinstance(result, (int, float)):
            raise ValueError("Calculation did not result in a number.")
            
        return result

    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except Exception as e:
        # Catch any other errors during parsing or evaluation
        raise ValueError(f"Could not compute expression: {e}")

if __name__ == '__main__':
    # Example Usage:
    test_expressions = [
        ("1 + 1", 2),
        ("10 - 2 * 3", 4),
        ("(10 - 2) * 3", 24),
        ("10 / 2", 5.0),
        ("10 / 0", "ValueError"),
        ("1 + ", "ValueError"),
        ("1 + 'abc'", "ValueError"),
        ("import os", "ValueError"),
        ("print('hello')", "ValueError"),
        ("__import__('os').system('ls')", "ValueError"),
        ("2.5 * 4", 10.0),
        ("-5 + 2", -3),
        ("abs(-1)", "ValueError"), # abs is a Name node, disallowed
        ("1 + unknown_var", "ValueError") # unknown_var is a Name node, disallowed
    ]

    for expr, expected in test_expressions:
        try:
            result = parse_and_calculate(expr)
            print(f"Expression: '{expr}', Result: {result}, Expected: {expected}")
            if isinstance(expected, str) and expected == "ValueError":
                print("  FAILED: Expected ValueError but got result.")
            elif result != expected:
                print(f"  FAILED: Expected {expected} but got {result}.")
        except ValueError as e:
            print(f"Expression: '{expr}', Error: {e}, Expected: {expected}")
            if expected != "ValueError":
                print(f"  FAILED: Expected {expected} but got ValueError.")
        except Exception as e:
            print(f"Expression: '{expr}', UNEXPECTED Error: {e}")
