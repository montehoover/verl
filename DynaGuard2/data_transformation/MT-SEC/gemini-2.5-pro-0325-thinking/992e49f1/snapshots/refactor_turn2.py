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

    # --- AST Node Whitelisting ---
    # To prevent arbitrary code execution, we define a whitelist of allowed
    # AST node types and specific operators. Only nodes and operations
    # corresponding to simple arithmetic will be permitted.

    # Allowed AST node types. These are the fundamental building blocks
    # of the expression's structure that we permit.
    allowed_node_types = (
        ast.Expression,  # The root node for an expression.
        ast.Constant,    # For numeric literals (e.g., 5, 3.14).
        ast.BinOp,       # For binary operations (e.g., +, -, *, /).
        ast.UnaryOp,     # For unary operations (e.g., -5).
    )
    
    # Allowed binary operator types (used within ast.BinOp nodes).
    # These define the mathematical operations like +, -, *, etc.
    allowed_bin_op_types = (
        ast.Add,        # Addition (+)
        ast.Sub,        # Subtraction (-)
        ast.Mult,       # Multiplication (*)
        ast.Div,        # True division (/)
        ast.Pow,        # Exponentiation (**)
        ast.Mod,        # Modulo (%)
        ast.FloorDiv    # Floor division (//)
    )
    
    # Allowed unary operator types (used within ast.UnaryOp nodes).
    # These define operations like unary plus (+x) and unary minus (-x).
    allowed_unary_op_types = (
        ast.UAdd,       # Unary plus (+x)
        ast.USub        # Unary minus (-x)
    )

    # --- AST Traversal and Validation ---
    # Walk through the parsed AST and check each node against the whitelists.
    # This is crucial for security, ensuring no disallowed operations or
    # node types are present in the expression.
    for sub_node in ast.walk(node):
        # Check 1: Ensure the node itself is of an allowed type (e.g., Expression, Constant, BinOp, UnaryOp).
        if not isinstance(sub_node, allowed_node_types):
            raise ValueError(
                f"Unsafe operation: Node type '{type(sub_node).__name__}' is not allowed. "
                f"Operation: '{operation}'"
            )

        # Check 2: For constants, ensure they are numeric (integers or floats).
        # This prevents evaluation of other constant types like strings or None, if they were to appear.
        if isinstance(sub_node, ast.Constant):
            if not isinstance(sub_node.value, (int, float)):
                raise ValueError(
                    f"Invalid constant: Only numeric constants (int, float) are allowed. "
                    f"Found type '{type(sub_node.value).__name__}' for value '{sub_node.value}'. "
                    f"Operation: '{operation}'"
                )
        # Check 3: For binary operations, ensure the specific operator (e.g., Add, Sub) is allowed.
        elif isinstance(sub_node, ast.BinOp):
            if not isinstance(sub_node.op, allowed_bin_op_types):
                raise ValueError(
                    f"Unsafe binary operator: Operator type '{type(sub_node.op).__name__}' is not allowed. "
                    f"Operation: '{operation}'"
                )
        # Check 4: For unary operations, ensure the specific operator (e.g., UAdd, USub) is allowed.
        elif isinstance(sub_node, ast.UnaryOp):
            if not isinstance(sub_node.op, allowed_unary_op_types):
                raise ValueError(
                    f"Unsafe unary operator: Operator type '{type(sub_node.op).__name__}' is not allowed. "
                    f"Operation: '{operation}'"
                )

    # --- Safe Evaluation ---
    # If all AST nodes and operations have been validated, proceed to compile and
    # evaluate the expression in a restricted environment.
    try:
        # Compile the AST Expression node into a code object.
        # 'node' is an ast.Expression as returned by ast.parse(..., mode='eval').
        # The actual expression content is in node.body, but compile handles the Expression node directly.
        code_object = compile(node, filename='<string>', mode='eval')
        
        # Evaluate the compiled code.
        # Provide empty dictionaries for globals and locals to restrict the execution environment,
        # preventing access to built-in functions or other potentially unsafe operations.
        result = eval(code_object, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        raise ValueError(f"Error during evaluation: Division by zero. Operation: '{operation}'")
    except Exception as e:
        # Catch any other errors during evaluation (e.g., overflow).
        raise ValueError(f"Error during evaluation of '{operation}': {e}")
