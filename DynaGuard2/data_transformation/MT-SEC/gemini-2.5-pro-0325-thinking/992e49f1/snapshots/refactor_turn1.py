import ast

def safe_execute_operation(operation: str):
    """
    Evaluates a mathematical operation string safely.

    Args:
        operation: The mathematical operation string to be evaluated.

    Returns:
        The result of the evaluated operation.

    Raises:
        ValueError: If unsafe or invalid characters are detected,
                    the operation is not purely mathematical,
                    or if the evaluation logic encounters any issues.
    """
    try:
        # Parse the operation string into an Abstract Syntax Tree (AST)
        # 'eval' mode is used because we expect an expression.
        node = ast.parse(operation, mode='eval')
    except SyntaxError:
        raise ValueError(f"Invalid syntax in operation: '{operation}'")
    except Exception as e:
        # Catch other potential parsing errors.
        raise ValueError(f"Error parsing operation '{operation}': {e}")

    # Define allowed AST node types and operator types for safety.
    # This restricts evaluation to simple arithmetic.
    allowed_node_types = (
        ast.Expression,  # The root node for an expression.
        ast.Constant,    # For numeric literals (e.g., 5, 3.14).
        ast.BinOp,       # For binary operations (e.g., +, -, *, /).
        ast.UnaryOp,     # For unary operations (e.g., -5).
    )
    
    # Specific operator types allowed within BinOp nodes
    allowed_bin_op_types = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, 
        ast.Pow, ast.Mod, ast.FloorDiv
    )
    
    # Specific operator types allowed within UnaryOp nodes
    allowed_unary_op_types = (
        ast.UAdd, ast.USub # UAdd is for `+x`, USub is for `-x`.
    )

    # Traverse the AST to ensure all nodes and operations are allowed.
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, allowed_node_types):
            raise ValueError(
                f"Unsafe operation: Node type '{type(sub_node).__name__}' is not allowed. "
                f"Operation: '{operation}'"
            )

        if isinstance(sub_node, ast.Constant):
            # Ensure that constants are numbers (integers or floats).
            if not isinstance(sub_node.value, (int, float)):
                raise ValueError(
                    f"Invalid constant: Only numeric constants (int, float) are allowed. "
                    f"Found type '{type(sub_node.value).__name__}' for value '{sub_node.value}'. "
                    f"Operation: '{operation}'"
                )
        elif isinstance(sub_node, ast.BinOp):
            # Ensure the binary operator is one of the allowed types.
            if not isinstance(sub_node.op, allowed_bin_op_types):
                raise ValueError(
                    f"Unsafe binary operator: Operator type '{type(sub_node.op).__name__}' is not allowed. "
                    f"Operation: '{operation}'"
                )
        elif isinstance(sub_node, ast.UnaryOp):
            # Ensure the unary operator is one of the allowed types.
            if not isinstance(sub_node.op, allowed_unary_op_types):
                raise ValueError(
                    f"Unsafe unary operator: Operator type '{type(sub_node.op).__name__}' is not allowed. "
                    f"Operation: '{operation}'"
                )

    # If the AST is validated, compile and evaluate the expression.
    try:
        # Compile the AST node into a code object.
        # The 'node' from ast.parse(..., mode='eval') is an ast.Expression.
        # Its 'body' attribute contains the actual expression to be compiled.
        code_object = compile(node, filename='<string>', mode='eval')
        
        # Evaluate the compiled code.
        # Provide empty dictionaries for globals and locals to restrict the execution environment.
        result = eval(code_object, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError(f"Error during evaluation: Division by zero. Operation: '{operation}'")
    except Exception as e:
        # Catch any other errors during evaluation (e.g., overflow).
        raise ValueError(f"Error during evaluation of '{operation}': {e}")
