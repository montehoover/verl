import ast

# Helper class to validate AST nodes
class SafeEvaluator(ast.NodeVisitor):
    # Define allowed AST node types for basic arithmetic and string operations
    ALLOWED_NODES = {
        ast.Expression,  # Root node for 'eval' mode
        ast.Constant,    # For numbers, strings, True, False, None (Python 3.8+)
        ast.BinOp,       # For binary operations (e.g., +, -, *, /)
        ast.UnaryOp,     # For unary operations (e.g., - negation)
        
        # Specific operator types allowed within BinOp/UnaryOp
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv, # Binary operators
        ast.USub, ast.UAdd, # Unary operators
    }

    # For Python versions older than 3.8, True, False, None were ast.NameConstant
    if hasattr(ast, 'NameConstant'):
        ALLOWED_NODES.add(ast.NameConstant)

    def __init__(self):
        super().__init__()
        self.safe = True

    def generic_visit(self, node):
        # If already determined to be unsafe, no need to check further
        if not self.safe:
            return

        # Check if the current node type is in the allowed set
        if type(node) not in self.ALLOWED_NODES:
            self.safe = False
            return
        
        # Continue visiting child nodes
        super().generic_visit(node)

def process_user_code(code_script: str) -> str:
    """
    Securely parses and executes a Python script string containing basic
    arithmetic or string operations.

    Args:
        code_script: A Python script string.

    Returns:
        The evaluated result as a string, or 'Execution Blocked!'
        if the script is unsafe or causes an error.
    """
    try:
        # Parse the user's code string into an AST.
        # 'eval' mode is used because we expect an expression.
        parsed_ast = ast.parse(code_script, mode='eval')
    except SyntaxError:
        # If parsing fails (e.g., invalid Python syntax), block execution.
        return "Execution Blocked!"

    # Initialize and run the AST validator
    evaluator = SafeEvaluator()
    evaluator.visit(parsed_ast)

    if not evaluator.safe:
        # If the validator found any disallowed AST nodes, block execution.
        return "Execution Blocked!"

    try:
        # Compile the validated AST into a code object.
        # '<user_script>' is a conventional name for the source of dynamic code.
        code_obj = compile(parsed_ast, filename='<user_script>', mode='eval')
        
        # Execute the compiled code object.
        # Pass extremely restricted globals: only an empty __builtins__ dict.
        # This prevents access to almost all built-in functions and names.
        # Pass empty locals.
        result = eval(code_obj, {"__builtins__": {}}, {})
        
        # Return the result, converted to a string as per requirements.
        return str(result)
    except Exception:
        # If any error occurs during compilation or evaluation (e.g., ZeroDivisionError),
        # or if a theoretically "safe" AST still causes issues, block execution.
        return "Execution Blocked!"
