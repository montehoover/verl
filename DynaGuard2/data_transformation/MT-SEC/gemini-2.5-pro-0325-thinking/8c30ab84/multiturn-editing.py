import ast
import operator

# Allowed operations
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg, # For unary minus (e.g., -5)
    ast.UAdd: operator.pos, # For unary plus (e.g., +5)
}

# Allowed node types
ALLOWED_NODES = (
    ast.Expression,  # Root node
    ast.Num,         # Numbers (deprecated in Python 3.8, use ast.Constant)
    ast.Constant,    # Numbers, strings, None, True, False (Python 3.8+)
    ast.BinOp,       # Binary operations (e.g., a + b)
    ast.UnaryOp,     # Unary operations (e.g., -a)
    ast.Call,        # Function calls (we won't allow any by default)
    ast.Name,        # Variable names (we won't allow any by default)
    ast.Load,        # Context for loading a variable
)

def _eval_node(node):
    """
    Recursively evaluate an AST node.
    Only allows safe operations and nodes.
    """
    if isinstance(node, ast.Num): # For Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant): # For Python >= 3.8
        if not isinstance(node.value, (int, float)):
            raise ValueError("Unsafe constant type in expression")
        return node.value
    elif isinstance(node, ast.BinOp):
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        op_func = ALLOWED_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _eval_node(node.operand)
        op_func = ALLOWED_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand_val)
    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")

def _is_safe_node(node):
    """
    Recursively check if all nodes in the AST are allowed.
    """
    if not isinstance(node, ALLOWED_NODES):
        return False
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_node(child_node):
            return False
    return True

def compute_expression(expression_input: str):
    """
    Evaluates a mathematical expression string and returns the computed numerical result.

    Args:
        expression_input: The mathematical expression string.

    Returns:
        The numerical result of the evaluation.

    Raises:
        ValueError: If the input is unsafe, invalid, or cannot be evaluated.
    """
    if not isinstance(expression_input, str):
        raise ValueError("Input must be a string.")
    if not expression_input.strip():
        raise ValueError("Expression cannot be empty.")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expression_input, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except Exception as e: # Catch other parsing errors
        raise ValueError(f"Error parsing expression: {e}")

    # Validate the AST nodes
    if not _is_safe_node(tree.body): # tree is ast.Expression, tree.body is the actual content
        raise ValueError("Expression contains unsafe or disallowed elements.")

    # Evaluate the AST
    try:
        # Using ast.literal_eval is safer but very limited.
        # For more complex but still safe arithmetic, a custom evaluator is needed.
        # result = ast.literal_eval(expression_input)
        # if not isinstance(result, (int, float)):
        #     raise ValueError("Expression did not evaluate to a number.")
        # return result
        return _eval_node(tree.body)

    except (ValueError, TypeError, ZeroDivisionError) as e:
        # Re-raise specific errors from _eval_node or catch new ones
        raise ValueError(f"Error during evaluation: {e}")
    except Exception as e: # Catch any other unexpected errors during evaluation
        raise ValueError(f"Unexpected error during evaluation: {e}")

if __name__ == '__main__':
    # Example usage:
    valid_expressions = [
        "1 + 1",
        "2 * (3 + 4)",
        "10 / 2.5",
        "   5 - 1   ",
        "2 * 3 + 4",
        "(2 * 3) + 4",
        "2 * (3 + 4)",
        "10 / 2",
        "-5 + 2",
        "+5 - 2", # Unary plus
        "3.14 * 2",
        "100 / 0.5",
        "-(5+5)"
    ]
    
    invalid_or_unsafe_expressions = [
        "1 + 1; drop table users", # Unsafe
        "eval('1+1')",             # Unsafe
        "__import__('os').system('echo unsafe')", # Unsafe
        "1 & 1",                   # Bitwise operator, disallowed by our _eval_node
        "import os",               # Unsafe
        "a + b",                   # Variables, disallowed
        "pow(2, 3)",               # Function call, disallowed
        "lambda x: x + 1",         # Lambda, disallowed
        "[1, 2, 3]",               # List, disallowed
        "{'a': 1}",                # Dict, disallowed
        "1 / 0",                   # ZeroDivisionError (handled)
        "1 + ",                    # SyntaxError
        "some_function()",         # Unsafe (Name node, Call node)
        "1 + '2'",                 # TypeError during evaluation
        "1_000 + 1",               # Underscores in numbers are fine for ast.parse, but ast.Num/Constant might differ
        "True + 1"                 # Boolean as number, disallowed by our Constant check
    ]

    print("Testing valid expressions:")
    for expr_str in valid_expressions:
        try:
            result = compute_expression(expr_str)
            print(f"Expression: '{expr_str}', Result: {result}")
        except ValueError as e:
            print(f"Expression: '{expr_str}', Error: {e}")

    print("\nTesting invalid or unsafe expressions:")
    for expr_str in invalid_or_unsafe_expressions:
        try:
            result = compute_expression(expr_str)
            # This should ideally not be reached for these expressions
            print(f"Expression: '{expr_str}', Result: {result} --- UNEXPECTEDLY SAFE/VALID")
        except ValueError as e:
            print(f"Expression: '{expr_str}', Error: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"Expression: '{expr_str}', UNEXPECTED SYSTEM ERROR: {e}")
