import ast
import io
import sys
from contextlib import redirect_stdout

# Helper class for AST validation to detect forbidden operations
class SafeASTChecker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        # Define forbidden AST node types (e.g., import statements)
        self.forbidden_node_types = (
            ast.Import,
            ast.ImportFrom,
        )
        # Define names of functions that are forbidden to be called
        self.forbidden_call_names = {
            'eval',
            'exec',
            'open',
            '__import__',
            # 'getattr', 'setattr', 'delattr', # Could be too restrictive or need context
        }
        # Define attributes that are forbidden to be accessed
        self.forbidden_attributes = {
            '__builtins__',  # Direct access to the full builtins object
            '__globals__',
            '__class__',
            '__subclasses__',
            '__mro__',
            '__bases__',
            '__dict__', # Accessing __dict__ can be risky on certain objects
            'system', 'remove', 'listdir', # Common os module functions if os was accessible
            'popen', 'spawn', # Other dangerous functions
        }
        self.violations = []

    def visit(self, node):
        # Check if the node itself is of a forbidden type
        if isinstance(node, self.forbidden_node_types):
            self.violations.append(
                f"Forbidden statement type: {node.__class__.__name__} "
                f"at line {getattr(node, 'lineno', 'N/A')}"
            )
        super().visit(node) # Continue to visit child nodes

    def visit_Call(self, node):
        # Check for calls to forbidden functions
        func_name = None
        if isinstance(node.func, ast.Name):  # e.g., open(...)
            func_name = node.func.id
        # Add checks for ast.Attribute if needed, e.g., some_module.forbidden_func()
        # elif isinstance(node.func, ast.Attribute):
        #     if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os': # Example
        #         func_name = node.func.attr


        if func_name and func_name in self.forbidden_call_names:
            self.violations.append(
                f"Forbidden function call: {func_name} at line {node.lineno}"
            )
        self.generic_visit(node) # Ensure all child nodes of the call are visited

    def visit_Attribute(self, node):
        # Check for access to forbidden attributes
        # e.g., some_object.__class__ or ().__class__.__bases__[0]...
        if node.attr in self.forbidden_attributes:
            obj_repr = "object"
            try:
                # Try to get a simple representation of the object being accessed
                if isinstance(node.value, ast.Name):
                    obj_repr = node.value.id
                elif isinstance(node.value, ast.Constant):
                    obj_repr = repr(node.value.value)
                # For more complex expressions, ast.unparse (Py3.9+) or a similar utility could be used
                # but for now, a generic "object" or simple name/constant is sufficient.
            except Exception:
                pass # Keep obj_repr as "object"

            self.violations.append(
                f"Forbidden attribute access: '{node.attr}' on '{obj_repr}' at line {node.lineno}"
            )
        self.generic_visit(node) # Ensure child nodes of the attribute are visited


def manage_script(user_script: str) -> str:
    """
    Reformats a script by appending a custom comment, checks for suspicious keywords,
    enforces a maximum length, and validates script integrity.

    Args:
        user_script: The script string to process.

    Returns:
        The processed script string with "# Processed" appended.

    Raises:
        ValueError: If the script contains suspicious keywords ("exec", "eval"),
                    if the script exceeds 100 characters,
                    or if the script is malformed (SyntaxError).
    """
    MAX_SCRIPT_LENGTH = 100
    if len(user_script) > MAX_SCRIPT_LENGTH:
        raise ValueError(f"Script exceeds maximum length of {MAX_SCRIPT_LENGTH} characters.")

    suspicious_keywords = ["exec", "eval"]
    for keyword in suspicious_keywords:
        if keyword in user_script:
            raise ValueError(f"Script contains suspicious keyword: {keyword}")

    try:
        # Attempt to parse the script to check for basic well-formedness
        # compile() can also be used here, ast.parse is often preferred for static analysis
        ast.parse(user_script)
    except SyntaxError as e:
        raise ValueError(f"Script is malformed: {e}") from e

    processed_script = user_script.strip()
    if processed_script:  # Add newline only if script is not empty
        processed_script += "\n"
    processed_script += "# Processed"
    return processed_script


def execute_user_script(submitted_script: str):
    """
    Executes a user-submitted script safely after performing syntax and security checks.

    Args:
        submitted_script: The script string to execute.

    Returns:
        The standard output from the script as a string, or None if there's no output.

    Raises:
        ValueError: If the script has syntax errors, contains forbidden actions/operations,
                    or if an error occurs during its execution.
    """
    # 1. Syntax Check: Parse the script into an Abstract Syntax Tree (AST)
    try:
        tree = ast.parse(submitted_script, filename="<submitted_script>")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}") from e

    # 2. Forbidden Actions Check: Traverse the AST to find disallowed operations
    checker = SafeASTChecker()
    checker.visit(tree)
    if checker.violations:
        violations_summary = "; ".join(checker.violations)
        raise ValueError(f"Script contains forbidden actions: {violations_summary}")

    # 3. Safe Execution: Execute the script in a restricted environment
    # Define a whitelist of safe built-in functions and objects
    safe_builtins = {
        'print': print,
        'len': len, 'range': range, 'abs': abs, 'round': round,
        'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'None': None, 'True': True, 'False': False,
        'zip': zip, 'enumerate': enumerate,
        # Add other builtins if they are deemed safe and necessary
        # Be cautious: e.g., 'globals', 'locals', 'vars' can expose too much
    }

    # The script will only have access to these globals, including our custom builtins
    restricted_globals = {"__builtins__": safe_builtins}
    # Locals will be populated by the script itself
    restricted_locals = {}

    output_capture = io.StringIO()
    try:
        # Compile the AST to a code object for execution
        code_obj = compile(tree, filename="<submitted_script>", mode="exec")
        
        # Redirect stdout to capture print statements from the script
        with redirect_stdout(output_capture):
            exec(code_obj, restricted_globals, restricted_locals)
            
    except Exception as e:
        # Catch runtime errors from the executed script
        # Provide a generic error message to avoid leaking sensitive details
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}") from e

    output = output_capture.getvalue()
    
    # Return the captured output, or None if nothing was printed
    return output if output else None
