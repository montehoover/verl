import ast

def _is_safe_node(node):
    """
    Recursively checks if an AST node is safe for evaluation.
    Allows numbers, basic arithmetic binary operations (+, -, *, /, **),
    and unary operations (+, -).
    """
    if isinstance(node, ast.Expression):
        return _is_safe_node(node.body)
    elif isinstance(node, ast.Constant):
        # Allow numeric constants (integers and floats)
        return isinstance(node.value, (int, float))
    elif isinstance(node, ast.Num):  # For compatibility with older Python versions (pre-3.8)
        # Allow numeric constants (integers and floats)
        return isinstance(node.n, (int, float))
    elif isinstance(node, ast.BinOp):
        # Allow specific binary operators
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return False
        # Recursively check left and right operands
        return _is_safe_node(node.left) and _is_safe_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Allow specific unary operators
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        # Recursively check the operand
        return _is_safe_node(node.operand)
    
    # Disallow all other node types (e.g., ast.Name, ast.Call, ast.Attribute, etc.)
    return False

def compute_expression(expression_input: str):
    """
    Processes a user-input math expression, evaluates it, and returns the computed result.

    Args:
        expression_input: str, a string that contains the mathematical expression to be processed.

    Returns:
        The result of the mathematical computation (float or int).

    Raises:
        TypeError: If expression_input is not a string.
        ValueError: In case any illegal characters or unsafe usage is found in the expression,
                    or if the expression cannot be safely evaluated (e.g., syntax error,
                    unsafe operation, division by zero, overflow).
    """
    if not isinstance(expression_input, str):
        raise TypeError("Expression input must be a string.")

    if not expression_input.strip():
        raise ValueError("Expression input cannot be empty or only whitespace.")

    try:
        # Parse the expression into an AST node.
        # mode='eval' ensures it's a single expression, not statements.
        parsed_expression = ast.parse(expression_input, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except Exception as e: # Catch other potential ast.parse errors
        raise ValueError(f"Error parsing expression: {e}")

    # Validate the AST to ensure it only contains allowed operations/nodes
    if not _is_safe_node(parsed_expression):
        raise ValueError("Expression contains unsafe or disallowed operations.")

    try:
        # Compile the AST node into a code object.
        # The filename '<string>' is conventional for dynamically generated code.
        code = compile(parsed_expression, filename='<string>', mode='eval')
        
        # Evaluate the compiled code.
        # Provide empty dictionaries for globals and locals to restrict access to
        # built-in functions or variables, enhancing safety.
        result = eval(code, {"__builtins__": {}}, {})
        
        # Ensure result is a number, as ast.Constant could theoretically hold other types
        # if _is_safe_node was less strict. Current _is_safe_node ensures numeric constants.
        if not isinstance(result, (int, float)):
             # This case should ideally not be reached if _is_safe_node is correct
             # and only allows numeric constants.
            raise ValueError("Evaluation resulted in a non-numeric type.")
            
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except OverflowError:
        raise ValueError("Numerical result out of range.")
    except Exception as e:
        # Catch any other errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")
