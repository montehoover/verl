import ast

# Allowed AST node types for safe evaluation.
# We only allow expressions, constants, binary operations, and unary operations.
# This implicitly restricts to basic arithmetic and string operations when used with eval.
# ast.Constant is used for literals (numbers, strings, True, False, None) in Python 3.8+.
ALLOWED_NODE_TYPES = (
    ast.Expression,  # The root of an 'eval' mode AST must be an Expression node.
    ast.Constant,    # For literals like numbers, strings.
    ast.BinOp,       # For binary operations like +, -, *, /, //, %, **.
    ast.UnaryOp,     # For unary operations like -, +.
)

def evaluate_user_code(code_str: str) -> str:
    """
    Securely evaluates a user-supplied Python script string.
    Supports basic arithmetic and string manipulations.
    Returns the result as a string if safe, or 'Execution Blocked!' if unsafe or an error occurs.
    """
    try:
        # Parse the code string into an AST.
        # mode='eval' ensures it's a single expression.
        tree = ast.parse(code_str, mode='eval')
    except SyntaxError:
        # If parsing fails, the code is malformed or not a simple expression.
        return "Execution Blocked!"

    # Validate all nodes in the AST.
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            # If any node is not in our whitelist, block execution.
            # This prevents constructs like function calls (ast.Call), attribute access (ast.Attribute),
            # variable names (ast.Name), imports, statements (ast.Assign, ast.If, etc.).
            return "Execution Blocked!"

    try:
        # Compile the AST to a code object.
        # This is a necessary step before eval'ing an AST and can catch some errors.
        compiled_code = compile(tree, filename='<user_code>', mode='eval')

        # Evaluate the compiled code.
        # Provide very restricted globals and no locals to prevent access to builtins
        # or other parts of the execution environment.
        # The arithmetic and string operations themselves (+, -, *, / on constants)
        # are handled by eval based on the types of the operands.
        result = eval(compiled_code, {'__builtins__': {}}, {})
        
        # Return the result, converted to a string.
        return str(result)
    except Exception:
        # Catch any other exceptions during compilation or evaluation.
        # This includes runtime errors from allowed operations (e.g., ZeroDivisionError)
        # or potential issues with overly complex "safe" expressions (e.g., OverflowError).
        # As per the requirement, any failure to produce a result safely leads to "Execution Blocked!".
        return "Execution Blocked!"
