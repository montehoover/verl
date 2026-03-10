import ast
import types # Added for CodeType

# --- Helper for AST Safety Check (existing, kept for clarity) ---
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

# --- Pipeline Step 1: Validate Input String ---
def _validate_input_string(expression_input: str) -> str:
    """Validates the input string type and content."""
    if not isinstance(expression_input, str):
        raise TypeError("Expression input must be a string.")
    if not expression_input.strip():
        raise ValueError("Expression input cannot be empty or only whitespace.")
    return expression_input

# --- Pipeline Step 2: Parse Expression to AST ---
def _parse_expression_to_ast(expression_str: str) -> ast.AST:
    """Parses the expression string into an AST node."""
    try:
        # mode='eval' ensures it's a single expression, not statements.
        return ast.parse(expression_str, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except Exception as e: # Catch other potential ast.parse errors
        raise ValueError(f"Error parsing expression: {e}")

# --- Pipeline Step 3: Validate AST Safety ---
def _validate_ast_safety(parsed_expression: ast.AST) -> ast.AST:
    """Validates the AST for safe operations."""
    if not _is_safe_node(parsed_expression):
        raise ValueError("Expression contains unsafe or disallowed operations.")
    return parsed_expression

# --- Pipeline Step 4: Compile AST to Code Object ---
def _compile_ast_to_code(safe_ast: ast.AST) -> types.CodeType:
    """Compiles the validated AST node into a code object."""
    try:
        # The filename '<string>' is conventional for dynamically generated code.
        return compile(safe_ast, filename='<string>', mode='eval')
    except Exception as e: # Catch potential compile errors
        raise ValueError(f"Error compiling expression: {e}")

# --- Pipeline Step 5: Evaluate Compiled Code ---
def _evaluate_compiled_code(code_obj: types.CodeType):
    """Evaluates the compiled code and validates the result type."""
    try:
        # Provide empty dictionaries for globals and locals to restrict access.
        result = eval(code_obj, {"__builtins__": {}}, {})
        
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

# --- Main Function Orchestrating the Pipeline ---
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
    validated_string = _validate_input_string(expression_input)
    parsed_ast = _parse_expression_to_ast(validated_string)
    safe_ast = _validate_ast_safety(parsed_ast)
    compiled_code = _compile_ast_to_code(safe_ast)
    result = _evaluate_compiled_code(compiled_code)
    return result
