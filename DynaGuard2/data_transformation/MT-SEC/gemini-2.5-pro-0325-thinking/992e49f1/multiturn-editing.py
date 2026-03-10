import ast

# Allowed operations and names for safe evaluation
ALLOWED_NAMES = {"None": None, "True": True, "False": False}
ALLOWED_NODE_TYPES = {
    'Expression', 'Num', 'BinOp', 'UnaryOp', 'NameConstant', 'Load',
    'Add', 'Sub', 'Mult', 'Div', 'Pow', 'USub', 'UAdd'
}

def is_safe_node(node):
    """
    Recursively check if an AST node is safe for evaluation.
    Only allows basic arithmetic operations and numbers.
    """
    if isinstance(node, ast.Num): # Numbers are safe
        return True
    elif isinstance(node, ast.Expression): # Expression body must be safe
        return is_safe_node(node.body)
    elif isinstance(node, ast.BinOp): # Binary operations: left, op, right must be safe
        return (type(node.op).__name__ in ALLOWED_NODE_TYPES and
                is_safe_node(node.left) and
                is_safe_node(node.right))
    elif isinstance(node, ast.UnaryOp): # Unary operations: op, operand must be safe
        return (type(node.op).__name__ in ALLOWED_NODE_TYPES and
                is_safe_node(node.operand))
    elif isinstance(node, ast.NameConstant): # True, False, None
        return node.value in ALLOWED_NAMES.values()
    # Disallow anything else (like Call, Name, Attribute, Subscript, etc.)
    return False


def safe_execute_operation(operation: str) -> float:
    """
    Evaluates a mathematical operation string safely.

    Args:
        operation: A string representing the mathematical operation, e.g., "5 + 3 * (2 - 1)".

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the operation string contains unsafe characters,
                    is syntactically incorrect, or evaluation fails.
    """
    if not isinstance(operation, str):
        raise ValueError("Operation must be a string.")

    # Basic check for obviously unsafe characters (though AST parsing is the main defense)
    # This is a pre-filter, ast.parse will do the heavy lifting for syntax.
    # We are mostly concerned about function calls, attribute access, etc.
    # A more robust way is to check the AST nodes.
    if any(char in operation for char in "();[]{}@_"): # Simplified check
        # More complex checks could involve regex for patterns like function calls `name(...)`
        # or attribute access `object.attribute`.
        # However, relying on AST node whitelisting is more robust.
        pass # Let AST parsing handle more complex syntax issues.

    try:
        # Parse the expression into an AST (Abstract Syntax Tree)
        # mode='eval' ensures it's an expression, not statements
        node = ast.parse(operation, mode='eval')

        # Validate the AST to ensure it only contains allowed operations/nodes
        if not is_safe_node(node):
            raise ValueError("Operation contains unsafe or disallowed elements.")

        # Compile the AST into a code object
        # <string> is a dummy filename for error messages
        code = compile(node, '<string>', 'eval')
        
        # Evaluate the code object
        # The __builtins__ dictionary is restricted to prevent access to unsafe built-in functions.
        # ALLOWED_NAMES provides a minimal set of safe global names (like True, False, None).
        result = eval(code, {"__builtins__": {}}, ALLOWED_NAMES)
        
        if not isinstance(result, (int, float)):
            raise ValueError("Evaluation did not result in a number.")
            
        return float(result)

    except SyntaxError:
        raise ValueError("Invalid syntax in operation string.")
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Error evaluating operation: {e}")
