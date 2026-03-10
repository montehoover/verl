import ast

# Allowed AST node types for the expression structure.
# ast.Num is included for compatibility with Python versions < 3.8.
# ast.Constant is preferred for Python 3.8+.
ALLOWED_NODE_TYPES = {
    ast.Expression,    # The root node of an expression.
    ast.Constant,      # For numeric literals (e.g., 5, 3.14) in Python 3.8+.
    ast.Num,           # For numeric literals (e.g., 5, 3.14) in Python < 3.8.
    ast.BinOp,         # Binary operations (e.g., a + b).
    ast.UnaryOp,       # Unary operations (e.g., -a).
}

# Allowed operator types for ast.BinOp (binary operations) and ast.UnaryOp (unary operations).
ALLOWED_OPERATOR_TYPES = {
    ast.Add,           # Addition (+)
    ast.Sub,           # Subtraction (-)
    ast.Mult,          # Multiplication (*)
    ast.Div,           # Division (/)
    ast.UAdd,          # Unary plus (+)
    ast.USub,          # Unary minus (-)
}

def _is_safe_ast_node(node):
    """
    Recursively checks if an AST node and its children are safe for evaluation.
    A node is considered safe if its type is in ALLOWED_NODE_TYPES.
    If it's an operator node (BinOp, UnaryOp), its specific operator type must
    be in ALLOWED_OPERATOR_TYPES.
    If it's a Constant/Num node, its value must be a number (int or float).
    """
    node_type = type(node)

    if node_type not in ALLOWED_NODE_TYPES:
        return False

    if isinstance(node, ast.Constant):
        # For ast.Constant (Python 3.8+), ensure the value is a number.
        if not isinstance(node.value, (int, float)):
            return False
    elif isinstance(node, ast.Num):
        # For ast.Num (Python < 3.8), ensure the value (n) is a number.
        if not isinstance(node.n, (int, float)):
            return False
    elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
        # For binary or unary operations, ensure the operator itself is allowed.
        if type(node.op) not in ALLOWED_OPERATOR_TYPES:
            return False
    
    # Recursively check all child nodes of the current node.
    for child_node in ast.iter_child_nodes(node):
        if not _is_safe_ast_node(child_node):
            return False
            
    return True

def evaluate_expression(expr_string: str) -> str:
    """
    Evaluates a user-provided mathematical expression string.

    The function parses the expression into an Abstract Syntax Tree (AST),
    validates that only basic arithmetic operations and numbers are used,
    and then evaluates the expression in a restricted environment.

    Args:
        expr_string: A string containing a basic mathematical expression.

    Returns:
        A string representing the result of a successful evaluation,
        or 'Computation Error!' if the input is suspicious, unsafe,
        malformed, or if the evaluation process fails for any reason
        (e.g., division by zero, overflow).
    """
    if not isinstance(expr_string, str):
        # Ensure input is a string.
        return "Computation Error!"
    
    if not expr_string.strip():
        # Handle empty or whitespace-only strings.
        return "Computation Error!"

    try:
        # Step 1: Parse the expression string into an AST.
        # 'eval' mode ensures that the string is parsed as a single expression.
        parsed_ast = ast.parse(expr_string, mode='eval')

        # Step 2: Validate the AST structure for safety.
        # This checks if all nodes and operations are within the allowed set.
        if not _is_safe_ast_node(parsed_ast):
            return "Computation Error!"

        # Step 3: Compile the validated AST into a code object.
        # The '<string>' filename is conventional for dynamically generated code.
        code_object = compile(parsed_ast, filename='<string>', mode='eval')
        
        # Step 4: Evaluate the compiled code object in a restricted environment.
        # Globals: {'__builtins__': {}} effectively removes access to all built-in functions.
        # Locals: An empty dictionary, providing no local variables.
        result = eval(code_object, {'__builtins__': {}}, {})

        # Step 5: Ensure the result is a number (int or float).
        # While AST validation should guarantee this, it's a good final check.
        if not isinstance(result, (int, float)):
            # This case should ideally not be reached if AST validation is robust.
            return "Computation Error!"

        return str(result)

    except (SyntaxError, TypeError, NameError, ValueError, ZeroDivisionError, OverflowError, MemoryError):
        # Catch common errors that can occur during parsing or evaluation
        # of expressions that might seem structurally valid but are problematic
        # (e.g., "1/0", "1+*2", numbers too large for memory/float precision).
        return "Computation Error!"
    except Exception:
        # Catch any other unexpected exceptions during the process.
        # This serves as a fallback for anything not caught by AST validation
        # or the more specific exceptions listed above.
        return "Computation Error!"
