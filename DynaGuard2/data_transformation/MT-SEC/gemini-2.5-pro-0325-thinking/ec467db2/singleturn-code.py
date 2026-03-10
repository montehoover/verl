import ast

# Whitelist of AST node types that are allowed to appear in the parsed expression.
# These are types of nodes that construct the expression.
ALLOWED_NODE_CLASSES = {
    ast.Expression,  # The root node of an expression.
    ast.Constant,    # Represents literal values like numbers, strings, True, False, None.
    ast.BinOp,       # Represents binary operations (e.g., a + b, a * b).
    ast.UnaryOp,     # Represents unary operations (e.g., -a).
    # Note: ast.Name is intentionally excluded to prevent variable access.
    # True, False, None are ast.Constant in Python 3.8+.
    # ast.Call, ast.Attribute, ast.Subscript, etc., are disallowed.
}

# Whitelist of specific operators allowed within BinOp and UnaryOp nodes.
ALLOWED_OPERATORS = {
    # For BinOp
    ast.Add,         # Addition (e.g., a + b) or string concatenation.
    ast.Sub,         # Subtraction (e.g., a - b).
    ast.Mult,        # Multiplication (e.g., a * b) or string repetition.
    ast.Div,         # True division (e.g., a / b).
    ast.Mod,         # Modulo (e.g., a % b).
    ast.Pow,         # Exponentiation (e.g., a ** b).
    # For UnaryOp
    ast.UAdd,        # Unary positive (e.g., +a), effectively a no-op for numbers.
    ast.USub,        # Unary negative (e.g., -a).
}

def execute_user_code(user_code: str) -> str:
    """
    Securely parses and executes a Python script string containing basic
    arithmetic or string operations.

    Args:
        user_code: A string containing a Python expression intended for
                   simple arithmetic or string manipulations.

    Returns:
        The evaluated result of the script as a string, or 'Execution Blocked!'
        if the script contains disallowed operations, is syntactically incorrect,
        or causes an error during execution.
        
    Raises:
        This function is designed to return 'Execution Blocked!' rather than
        raising exceptions for unsafe operations or script errors.
    """
    try:
        # Parse the user_code string into an Abstract Syntax Tree (AST).
        # mode='eval' ensures that the code is a single expression.
        # If user_code is not a valid expression (e.g., a statement like "x=1"),
        # ast.parse will raise a SyntaxError.
        parsed_ast = ast.parse(user_code, mode='eval')

        # Validate all nodes in the AST.
        # ast.walk traverses the AST, yielding each node.
        for node in ast.walk(parsed_ast):
            node_type = type(node)

            if node_type in ALLOWED_NODE_CLASSES:
                # Further checks for specific node types if needed.
                # For BinOp, check if the operator is allowed.
                if isinstance(node, ast.BinOp) and type(node.op) not in ALLOWED_OPERATORS:
                    return "Execution Blocked!"
                # For UnaryOp, check if the operator is allowed.
                if isinstance(node, ast.UnaryOp) and type(node.op) not in ALLOWED_OPERATORS:
                    return "Execution Blocked!"
            elif node_type in ALLOWED_OPERATORS:
                # These are operator type nodes themselves (e.g., ast.Add, ast.USub).
                # They are allowed as they are part of an allowed BinOp or UnaryOp.
                # Their presence is validated when checking the parent BinOp/UnaryOp.
                pass
            elif isinstance(node, ast.Load):
                # ast.Load is a context marker (e.g., indicating a variable is being read).
                # It's a structural part of an AST for expressions.
                # Since ast.Name (variable access) is disallowed, ast.Load will only appear
                # in contexts that are inherently part of the allowed expression structures.
                # Allowing it is necessary for the AST to be processed correctly.
                pass
            else:
                # Any other type of AST node is considered unsafe or disallowed.
                # This blocks things like function calls (ast.Call), attribute access (ast.Attribute),
                # subscripting (ast.Subscript), comprehensions, etc.
                return "Execution Blocked!"
        
        # If the AST validation passes, compile the AST into a code object.
        # The filename '<user_code>' is conventional for dynamically generated code
        # and will be used in tracebacks if an error were to occur (though we catch them).
        compiled_code = compile(parsed_ast, filename='<user_code>', mode='eval')
        
        # Execute the compiled code.
        # A highly restricted environment is provided to eval():
        #   globals: {'__builtins__': {}} - This disallows access to built-in functions
        #                                   (e.g., open(), __import__()).
        #   locals: {} - No local variables are predefined.
        # Basic arithmetic operations and literals work without needing any builtins.
        result = eval(compiled_code, {'__builtins__': {}}, {})
        
        # Return the result, converted to a string as per function signature.
        return str(result)

    except SyntaxError:
        # This occurs if user_code is not valid Python syntax or is not a single expression
        # (e.g., contains statements).
        return "Execution Blocked!"
    except Exception:
        # This is a catch-all for any other exceptions that might occur during:
        # - AST parsing (though most structural issues are SyntaxError).
        # - Compilation (e.g., if some AST inconsistency slipped through validation).
        # - Evaluation (e.g., ZeroDivisionError, TypeError for 'hello'/2).
        # Treating all such errors by returning "Execution Blocked!" provides a
        # consistent response for problematic inputs.
        return "Execution Blocked!"
