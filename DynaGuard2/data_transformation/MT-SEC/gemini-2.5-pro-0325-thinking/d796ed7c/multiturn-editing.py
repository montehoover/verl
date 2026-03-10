import ast
from typing import Union, Any

# Allowed AST node types for safe evaluation
ALLOWED_NODES = {
    ast.Expression,
    ast.Constant,  # For numbers (int, float)
    ast.Num,       # For numbers (older Python, ast.Constant is preferred)
    ast.BinOp,     # For binary operations like +, -, *, /
    ast.UnaryOp,   # For unary operations like - (negation)
    ast.Add,       # Specific operator type for +
    ast.Sub,       # Specific operator type for -
    ast.Mult,      # Specific operator type for *
    ast.Div,       # Specific operator type for /
    ast.USub,      # Specific operator type for unary -
    ast.UAdd,      # Specific operator type for unary + (usually optimized out but good to list)
}

def _is_node_allowed(node: ast.AST) -> bool:
    """Check if all nodes in the AST are allowed."""
    for n in ast.walk(node):
        if type(n) not in ALLOWED_NODES:
            return False
    return True

def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluates an AST node.
    Supports ast.Constant (for numbers), ast.BinOp (+, -, *, /), and ast.UnaryOp (USub, UAdd).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.Num): # For compatibility with older Python ASTs
        if isinstance(node.n, (int, float)):
            return node.n
        else:
            raise ValueError(f"Unsupported number type: {type(node.n).__name__}")
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        # This case should ideally be caught by _is_node_allowed before _eval_ast_node is called on an unallowed node type.
        # However, as a safeguard:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def secure_math_eval(exp_str: str) -> float:
    """
    Evaluates a mathematical expression string securely.

    Args:
        exp_str: The mathematical expression string (e.g., "5 + 7 * (3 - 1) / 2").

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the expression contains invalid characters, unsafe operations,
                    is malformed, or results in an error like division by zero.
        SyntaxError: If the expression string is not valid Python syntax.
    """
    if not isinstance(exp_str, str):
        raise TypeError("Expression must be a string.")

    try:
        # Parse the expression string into an AST. 'eval' mode for a single expression.
        # The parsed tree's root will be ast.Expression, whose body is the actual expression.
        parsed_ast = ast.parse(exp_str, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {exp_str}. Details: {e}")
    except Exception as e: # Catch other parsing related errors
        raise ValueError(f"Error parsing expression: '{exp_str}'. Details: {e}")

    # Validate that all nodes in the AST are allowed
    if not _is_node_allowed(parsed_ast):
        # Find the first disallowed node for a more specific error (optional)
        for node in ast.walk(parsed_ast):
            if type(node) not in ALLOWED_NODES:
                raise ValueError(
                    f"Expression contains disallowed operation or construct: {type(node).__name__}"
                )
        # Fallback generic error if specific node not found (should not happen if _is_node_allowed is False)
        raise ValueError("Expression contains disallowed operations or constructs.")

    # Evaluate the body of the expression (e.g., the BinOp node for "1+2")
    # The root of an 'eval' mode parse is ast.Expression, its .body is the actual expression.
    try:
        result = _eval_ast_node(parsed_ast.body)
        if not isinstance(result, (int, float)):
            # This should not happen if _eval_ast_node is correct
            raise ValueError("Evaluation did not result in a number.")
        return float(result)
    except ValueError: # Re-raise ValueErrors from _eval_ast_node (e.g., div by zero)
        raise
    except Exception as e: # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: '{exp_str}'. Details: {e}")

if __name__ == '__main__':
    # Example Usage:
    expressions = [
        "5 + 7",
        "10 - 2.5",
        "3 * 4",
        "10 / 2",
        "2 * (3 + 4)",
        "-5 + 10",
        "+2 - -3", # Unary plus and double negative
        "10 / 0", # Division by zero
        "10 / (2 - 2)", # Division by zero through sub-expression
        "2 ** 3", # Power operator (disallowed by current ALLOWED_NODES)
        "import os", # Disallowed
        "__import__('os').system('clear')", # Disallowed
        "a + 5", # Disallowed (Name 'a' is not a number)
        "eval('1+1')", # Disallowed
        "1; 2", # Disallowed (multiple statements)
        "print(5)", # Disallowed
        "abs(-5)", # Disallowed
    ]

    for expr in expressions:
        try:
            result = secure_math_eval(expr)
            print(f"'{expr}' => {result}")
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error evaluating '{expr}': {e}")

    print("\nTesting direct calls to _eval_ast_node (for dev/debug):")
    try:
        # Test case for ast.Num (if Python version makes ast.parse use it for numbers)
        # or ast.Constant
        node_num = ast.parse("123", mode='eval').body
        print(f"Evaluating node for '123': {_eval_ast_node(node_num)}")

        node_add = ast.parse("1+2", mode='eval').body
        print(f"Evaluating node for '1+2': {_eval_ast_node(node_add)}")
        
        # Example of a disallowed node type if directly passed (though _is_node_allowed should prevent this path)
        # node_name = ast.Name(id='x', ctx=ast.Load())
        # print(f"Evaluating ast.Name node (should fail): {_eval_ast_node(node_name)}")

    except Exception as e:
        print(f"Error in direct _eval_ast_node test: {e}")
