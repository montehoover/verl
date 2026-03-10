import ast

# Whitelist of allowed AST node types that can appear in a safe expression
# ast.Expression: The root node of an expression.
# ast.Constant: Represents literal values like numbers. We only allow numbers.
# ast.Num: Represents numbers (used in Python < 3.8, ast.Constant is preferred now).
# ast.BinOp: Represents binary operations (e.g., +, -, *, /).
# ast.UnaryOp: Represents unary operations (e.g., - for negation).
ALLOWED_NODE_TYPES = {
    ast.Expression,
    ast.Constant,
    ast.Num,  # For Python < 3.8 compatibility
    ast.BinOp,
    ast.UnaryOp,
}

# Whitelist of allowed binary operators
ALLOWED_BINARY_OPERATORS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.FloorDiv,
}

# Whitelist of allowed unary operators
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd, ast.USub,
}

def safe_math_evaluator(expr_string: str) -> str:
    """
    Safely evaluates a Python mathematical expression string.

    Args:
        expr_string: The string containing the mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Unsafe Code Detected!'
        if the expression is not safe or evaluation fails.
    """
    try:
        # 1. Parse the expression string into an Abstract Syntax Tree (AST).
        #    mode='eval' is crucial as it expects a single expression.
        tree = ast.parse(expr_string, mode='eval')

        # 2. Validate all nodes in the AST.
        for node in ast.walk(tree):
            node_type = type(node)

            if node_type not in ALLOWED_NODE_TYPES:
                # If the node's type is not in our whitelist, it's disallowed.
                # This catches things like ast.Call, ast.Name (variables/functions),
                # ast.Attribute, ast.Subscript, ast.ListComp, etc.
                return 'Unsafe Code Detected!'

            # Additional checks for specific node types:
            if node_type is ast.Constant:
                # For ast.Constant, ensure the value is a number (int or float).
                # ast.Constant can also represent strings, None, bools, bytes.
                if not isinstance(node.value, (int, float)):
                    return 'Unsafe Code Detected!'
            elif node_type is ast.Num:  # For Python < 3.8
                # ast.Num always holds a number.
                if not isinstance(node.n, (int, float)):
                    # This case should ideally not be hit if it's a valid ast.Num node
                    return 'Unsafe Code Detected!'
            elif node_type is ast.BinOp:
                # For binary operations, ensure the operator is in our whitelist.
                if type(node.op) not in ALLOWED_BINARY_OPERATORS:
                    return 'Unsafe Code Detected!'
            elif node_type is ast.UnaryOp:
                # For unary operations, ensure the operator is in our whitelist.
                if type(node.op) not in ALLOWED_UNARY_OPERATORS:
                    return 'Unsafe Code Detected!'

    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        # These errors can occur during parsing or AST walking if the expression
        # is malformed, too complex, or uses features not parsable in 'eval' mode.
        return 'Unsafe Code Detected!'
    except Exception:
        # Catch any other unexpected errors during parsing/validation.
        return 'Unsafe Code Detected!'

    # 3. If AST validation passes, compile and evaluate the expression.
    try:
        # Compile the AST into a code object.
        # filename='<expr_string>' is a convention for dynamically generated code.
        code = compile(tree, filename='<expr_string>', mode='eval')

        # As an additional security measure, check if the compiled code object
        # refers to any names (co_names). For purely mathematical expressions
        # with literals, this should be empty. This helps catch any ast.Name
        # nodes if they somehow bypassed the AST check.
        if code.co_names:
            return 'Unsafe Code Detected!'

        # Evaluate the compiled code in a restricted environment.
        # __builtins__ is emptied to prevent access to built-in functions.
        # Globals and locals are also empty.
        result = eval(code, {'__builtins__': {}}, {})
        
        # Ensure the result is a number before converting to string.
        if not isinstance(result, (int, float)):
            # This handles cases where an expression might evaluate to something
            # non-numeric, though this is unlikely with strict AST checks.
            return 'Unsafe Code Detected!'

        return str(result)
    
    except (ArithmeticError, TypeError, ValueError, NameError, ZeroDivisionError):
        # ArithmeticError: e.g., division by zero, overflow.
        # TypeError, ValueError: Mismatch in types during an operation.
        # NameError: Should not occur if co_names check is effective and ast.Name is disallowed.
        # ZeroDivisionError: A specific type of ArithmeticError.
        # As per requirements, evaluation failures also lead to 'Unsafe Code Detected!'.
        return 'Unsafe Code Detected!'
    except Exception:
        # Catch any other unexpected errors during compilation or evaluation.
        return 'Unsafe Code Detected!'
