import re
import ast

# Regex for allowed characters in the expression.
# Allows digits, whitespace, decimal points, and basic arithmetic operators: +, -, *, /, (, ).
# Note: '**' for power is handled by allowing '*' and ast.parse interpreting it correctly.
ALLOWED_CHAR_PATTERN = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

def _validate_node(node):
    """
    Recursively validates an AST node to ensure it's part of a safe, simple arithmetic expression.
    Raises ValueError if an unsupported construct is found.
    """
    node_type = type(node)

    if node_type == ast.Expression:
        # The root of an expression AST, validate its body.
        _validate_node(node.body)
    elif node_type == ast.BinOp:
        # Binary operations: +, -, *, /, **
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        _validate_node(node.left)
        _validate_node(node.right)
    elif node_type == ast.UnaryOp:
        # Unary operations: - (negation), + (unary plus)
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        _validate_node(node.operand)
    # Numeric constants: ast.Constant (Python 3.8+) or ast.Num (older Python)
    elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant): # Check for ast.Constant first
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant value: {node.value}. Only numbers (int, float) are allowed.")
    elif isinstance(node, ast.Num): # Fallback for ast.Num (used in Python < 3.8)
        if not isinstance(node.n, (int, float)):
            raise ValueError(f"Unsupported number value: {node.n}. Only numbers (int, float) are allowed.")
    else:
        # Any other AST node type is considered unsupported and unsafe.
        raise ValueError(f"Unsupported expression construct: {node_type.__name__}")

def evaluate_expression(expr: str):
    """
    Parses and evaluates a user-provided mathematical expression string.

    Args:
        expr: A string representing the arithmetic expression.

    Returns:
        The computed result of the expression (int or float).

    Raises:
        ValueError: If the expression contains unsupported characters,
                    unsafe commands, invalid operations, or results in an error
                    (e.g., syntax error, division by zero).
    """
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string.")

    stripped_expr = expr.strip()
    if not stripped_expr:
        raise ValueError("Expression cannot be empty.")

    # 1. Validate allowed characters
    if not ALLOWED_CHAR_PATTERN.match(stripped_expr):
        raise ValueError("Expression contains unsupported characters.")

    # 2. Parse the expression string into an AST
    try:
        # 'eval' mode ensures the string is parsed as an expression.
        parsed_ast = ast.parse(stripped_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")

    # 3. Validate the AST to ensure only allowed nodes and operations are present
    try:
        _validate_node(parsed_ast)
    except ValueError as e: # Catch validation errors from _validate_node
        raise ValueError(f"Invalid operation or construct in expression: {e}")

    # 4. Compile and evaluate the validated AST in a restricted environment
    try:
        # The `parsed_ast` is an ast.Expression node.
        # compile() expects this structure for 'eval' mode.
        code_object = compile(parsed_ast, filename="<string>", mode="eval")
        # Evaluate with empty globals and locals to restrict access to builtins and variables.
        result = eval(code_object, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except OverflowError:
        raise ValueError("Numerical result out of range.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        raise ValueError(f"Error during expression evaluation: {e}")

    # Ensure the result is a number, though _validate_node should guarantee this.
    if not isinstance(result, (int, float)):
        # This should ideally not be reached if AST validation is comprehensive.
        raise ValueError("Expression did not evaluate to a numerical result.")

    return result

if __name__ == '__main__':
    # Example Usage:
    test_expressions = [
        "3 + 5 * (2 - 1)",
        "10 / 2",
        "2 ** 3",
        "-5 + (2.5 * 4)",
        "1 / 0", # Expected: ValueError (Division by zero)
        "1 + '2'", # Expected: ValueError (Unsupported constant value or AST validation error)
        "abs(-5)", # Expected: ValueError (Unsupported characters or construct)
        "import os", # Expected: ValueError (Unsupported characters)
        "my_variable + 1" # Expected: ValueError (Unsupported characters or construct)
    ]

    for expr_str in test_expressions:
        try:
            output = evaluate_expression(expr_str)
            print(f"Input: \"{expr_str}\", Output: {output}")
        except (ValueError, TypeError) as e:
            print(f"Input: \"{expr_str}\", Error: {e}")

    # Example from problem description
    print("\nExample from problem description:")
    example_input = {"expr": "3 + 5 * (2 - 1)"}
    try:
        output = evaluate_expression(example_input["expr"])
        print(f"Input: {example_input}, Output: {output}") # Expected: 8
    except (ValueError, TypeError) as e:
        print(f"Input: {example_input}, Error: {e}")
