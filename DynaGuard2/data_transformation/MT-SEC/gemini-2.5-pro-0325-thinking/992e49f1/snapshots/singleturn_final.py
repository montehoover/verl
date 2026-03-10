import ast

# Define allowed AST node types for safe evaluation.
# These include the expression itself, constants (numbers),
# binary operations and their specific types (Add, Sub, etc.),
# and unary operations and their specific types (USub, UAdd).
_ALLOWED_NODES = {
    ast.Expression,  # Represents an entire expression
    ast.Constant,    # For numeric literals (Python 3.8+)
    ast.Num,         # For numeric literals (Python < 3.8)
    ast.BinOp,       # For binary operations like +, -, *, /
    ast.UnaryOp,     # For unary operations like - (negation)
    
    # Specific operator types (these are classes for the 'op' attribute
    # of BinOp and UnaryOp, and ast.walk visits them too)
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,    # For integer division //
    ast.USub,        # For unary minus
    ast.UAdd,        # For unary plus (e.g., +5)
}

def safe_execute_operation(operation: str):
    """
    Safely evaluates a mathematical operation string.

    Args:
        operation: The mathematical operation string to evaluate.

    Returns:
        The result of the evaluated operation.

    Raises:
        ValueError: If unsafe or invalid characters are detected,
                    or if the evaluation logic encounters any issues.
    """
    if not isinstance(operation, str):
        # Defensive type check, though type hints are present.
        raise TypeError("Operation must be a string.")

    if not operation.strip():
        raise ValueError("Operation string cannot be empty.")

    try:
        # Parse the operation string into an AST (Abstract Syntax Tree)
        # mode='eval' ensures it's an expression, not statements.
        node = ast.parse(operation, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in operation: {e}") from e
    except Exception as e: # Catch other ast.parse potential errors
        raise ValueError(f"Error parsing operation: {e}") from e

    # Validate all nodes in the AST
    for sub_node in ast.walk(node):
        if type(sub_node) not in _ALLOWED_NODES:
            raise ValueError(
                f"Unsafe operation: Disallowed construct '{type(sub_node).__name__}' found."
            )
        
        # Specifically for constants, ensure they are numbers
        if isinstance(sub_node, ast.Constant):
            # This also blocks strings, True/False, None as constants in the expression
            if not isinstance(sub_node.value, (int, float)):
                raise ValueError(
                    f"Unsafe operation: Non-numeric constant '{repr(sub_node.value)}' found."
                )
        elif isinstance(sub_node, ast.Num): # For Python < 3.8 compatibility
            if not isinstance(sub_node.n, (int, float)):
                raise ValueError(
                    f"Unsafe operation: Non-numeric constant '{repr(sub_node.n)}' found."
                )

    # If all nodes are safe, compile and evaluate the AST
    # The environment for eval is restricted to prevent access to builtins or global variables.
    try:
        # Compile the AST into a code object
        code = compile(node, filename='<operation_string>', mode='eval')
        # Evaluate the code object in a restricted environment
        # No access to builtins or any global/local scope variables.
        result = eval(code, {'__builtins__': {}}, {})
        return result
    except ZeroDivisionError as e:
        raise ValueError("Error during evaluation: Division by zero.") from e
    except OverflowError as e:
        raise ValueError("Error during evaluation: Numerical result out of range.") from e
    except Exception as e:
        # Catch any other errors during compilation or evaluation
        raise ValueError(f"Error during evaluation: {e}") from e
