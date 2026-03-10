import ast
import builtins
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Allowed built-in functions and types available to the snippet
ALLOWED_BUILTINS_NAMES = {
    'print', 'len', 'range', 'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr',
    'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
    'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter', 'list',
    'map', 'max', 'min', 'next', 'object', 'oct', 'ord', 'pow', 'repr', 'reversed',
    'round', 'set', 'slice', 'sorted', 'str', 'sum', 'super', 'tuple', 'type', 'zip',
    # Common exceptions (useful for try/except in snippet)
    'Exception', 'ValueError', 'TypeError', 'IndexError', 'KeyError', 'AttributeError', 'NameError',
    'ZeroDivisionError',
}

# Allowed modules that the snippet can import
ALLOWED_IMPORTS = {'math'}

# Function names that are disallowed if called directly by name
DISALLOWED_FUNCTION_CALLS = {
    'open', 'eval', 'exec', '__import__', 'compile', 'globals', 'locals', 'vars',
    'getattr', 'setattr', 'delattr', 'dir',
    # os/sys related functions often found as builtins or on modules
    'system', 'spawn', 'fork', 'kill', 'exit', 'quit'
}

# Attribute names that are disallowed from access or use
DISALLOWED_ATTRIBUTES = {
    '__builtins__', '__class__', '__subclasses__', '__dict__', '__globals__',
    '__code__', '__closure__', '__func__', '__self__', '__mro__', '__bases__',
    '__init__', '__new__', '__del__', '__enter__', '__exit__',
    # Some other common ones that might be risky
    'f_locals', 'f_globals', 'f_code', 'f_back', 'f_trace',
    'gi_frame', 'gi_code', # generator internals
    'co_code', 'co_consts', 'co_varnames' # code object internals
}

# Common dunder methods that are generally safe to call (e.g. str(obj), obj + other_obj)
ALLOWED_DUNDER_METHODS_CALL = {
    '__str__', '__repr__', '__len__', '__getitem__', '__setitem__', '__delitem__',
    '__contains__', '__iter__', '__next__',
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
    '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__hash__', '__bool__', '__call__'
}


class SafeVisitor(ast.NodeVisitor):
    """
    Traverses the AST to ensure no forbidden operations are attempted.
    Raises ValueError if an unsafe operation is detected.
    """
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                raise ValueError(f"Import of module '{alias.name}' is not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module not in ALLOWED_IMPORTS:
            raise ValueError(f"Import from module '{node.module}' is not allowed.")
        # Optionally, could check node.names for specific allowed names from the module
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # Direct call to a named function, e.g., open(...)
            if node.func.id in DISALLOWED_FUNCTION_CALLS:
                raise ValueError(f"Call to forbidden function '{node.func.id}' is not allowed.")
        elif isinstance(node.func, ast.Attribute):
            # Method call, e.g., some_object.dangerous_method()
            # Check the attribute name (method name)
            if node.func.attr in DISALLOWED_FUNCTION_CALLS:
                 raise ValueError(f"Call to forbidden method '{node.func.attr}' is not allowed.")
            # Check for calls to attributes starting with an underscore
            if node.func.attr.startswith('_') and node.func.attr not in ALLOWED_DUNDER_METHODS_CALL:
                 raise ValueError(f"Call to private or non-whitelisted dunder method '{node.func.attr}' is not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Accessing an attribute, e.g., some_object.dangerous_attribute
        if node.attr in DISALLOWED_ATTRIBUTES:
            raise ValueError(f"Access to forbidden attribute '{node.attr}' is not allowed.")
        
        # Prevent access to attributes starting with a single underscore (convention for protected)
        if node.attr.startswith('_') and not node.attr.startswith('__') and not node.attr.endswith('__'):
             raise ValueError(f"Access to protected attribute '{node.attr}' is not allowed.")
        
        # For dunder attributes not in DISALLOWED_ATTRIBUTES, allow access (e.g. for __add__ to be found)
        # but calls to them are checked in visit_Call.
        self.generic_visit(node)

    def visit_Exec(self, node: ast.Exec):
        raise ValueError("'exec' statement/function is not allowed within the snippet.")

    # Consider adding checks for ast.Delete, ast.Try (potential resource exhaustion), etc.
    # For now, this covers common high-risk areas.


def _parse_and_validate_snippet(snippet_string: str) -> ast.AST:
    """
    Parses the snippet string into an AST and validates it for safety.

    Args:
        snippet_string: The Python code to parse and validate.

    Returns:
        The validated Abstract Syntax Tree (AST) of the snippet.

    Raises:
        ValueError: If the snippet contains invalid syntax or attempts forbidden operations.
    """
    logger.debug(f"Attempting to parse and validate snippet: {snippet_string[:100]}{'...' if len(snippet_string) > 100 else ''}")
    try:
        # Parse the snippet into an AST. type_ignores is for Python 3.8+
        logger.debug("Parsing snippet string into AST.")
        tree = ast.parse(snippet_string, type_ignores=[])
        logger.debug("Snippet parsed successfully.")
    except SyntaxError as e:
        logger.warning(f"Syntax error during parsing: {e}")
        raise ValueError(f"Invalid Python code: {e}")

    # Validate the AST using the custom visitor
    validator = SafeVisitor()
    logger.debug("Validating AST.")
    try:
        validator.visit(tree)
        logger.info("AST validation successful.")
    except ValueError as e:
        logger.warning(f"AST validation failed: {e}")
        # Propagate the ValueError raised by the validator (contains specific error message)
        raise
    return tree


def _execute_validated_snippet(tree: ast.AST):
    """
    Executes a pre-validated AST in a restricted environment.

    Args:
        tree: The validated AST to execute.

    Returns:
        The result of the last expression in the snippet, or None if it ends with statements.

    Raises:
        ValueError: If a runtime error occurs during execution.
    """
    logger.debug("Preparing to execute validated AST.")
    # Prepare a restricted environment for execution
    actual_safe_builtins = {}
    for name in ALLOWED_BUILTINS_NAMES:
        if hasattr(builtins, name):
            actual_safe_builtins[name] = getattr(builtins, name)
    
    safe_globals = {"__builtins__": actual_safe_builtins}
    
    # Pre-populate globals with allowed modules if they are used by the snippet
    # The validator ensures only allowed modules can be imported.
    # Here, we make them available if they are imported.
    # For simplicity, if 'math' is allowed and imported, it will be available.
    # The snippet's 'import math' will work against this pre-populated global.
    if 'math' in ALLOWED_IMPORTS:
        import math as _math_module_for_sandbox # Import with an alias to avoid name collision
        safe_globals['math'] = _math_module_for_sandbox
        
    safe_locals = {} # Local scope for the execution

    # If the snippet's body is not empty and the last statement is an expression,
    # we try to evaluate it and return its result.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Compile all statements before the final expression
        statements_nodes = tree.body[:-1]
        if statements_nodes:
            module_statements = ast.Module(body=statements_nodes, type_ignores=[])
            ast.fix_missing_locations(module_statements) # Important for ASTs built/modified manually
            try:
                logger.debug("Compiling and executing preliminary statements.")
                code_obj_exec = compile(module_statements, '<snippet_statements>', 'exec')
                exec(code_obj_exec, safe_globals, safe_locals)
                logger.debug("Preliminary statements executed successfully.")
            except Exception as e:
                logger.warning(f"Error during execution of preliminary statements: {e}")
                raise ValueError(f"Error executing snippet statements: {e}")
        
        # Compile and evaluate the final expression
        expression_node = tree.body[-1].value
        expr_wrapper = ast.Expression(body=expression_node)
        ast.fix_missing_locations(expr_wrapper)
        logger.debug("Compiling and evaluating final expression.")
        try:
            code_obj_eval = compile(expr_wrapper, '<snippet_expression>', 'eval')
            result = eval(code_obj_eval, safe_globals, safe_locals)
            logger.info(f"Final expression evaluated successfully. Result: {result}")
            return result
        except Exception as e:
            logger.warning(f"Error during evaluation of final expression: {e}")
            raise ValueError(f"Error evaluating snippet expression: {e}")
            
    else:
        # The snippet is empty or does not end with an expression (e.g., ends with an assignment).
        # Execute the whole snippet as statements.
        ast.fix_missing_locations(tree)
        logger.debug("Compiling and executing snippet as statements (no final expression).")
        try:
            code_obj_exec = compile(tree, '<snippet_all_statements>', 'exec')
            exec(code_obj_exec, safe_globals, safe_locals)
            logger.info("Snippet executed successfully as statements. No return value.")
            return None # No specific result value for statements
        except Exception as e:
            logger.warning(f"Error during execution of snippet as statements: {e}")
            raise ValueError(f"Error executing snippet: {e}")


def run_python_snippet(snippet_string: str):
    """
    Executes a Python code snippet safely.

    Args:
        snippet_string: The Python code to execute.

    Returns:
        The result of the last expression in the snippet, or None if it ends with statements.

    Raises:
        TypeError: If snippet_string is not a string.
        ValueError: If the snippet contains invalid syntax, attempts forbidden operations,
                    or a runtime error occurs during execution.
    """
    logger.info(f"Received request to run snippet: {snippet_string[:100]}{'...' if len(snippet_string) > 100 else ''}")
    if not isinstance(snippet_string, str):
        logger.error(f"Type error: Snippet must be a string, got {type(snippet_string)}.")
        raise TypeError("Snippet must be a string.")

    try:
        # Step 1: Parse and validate the snippet
        # This can raise ValueError for syntax or validation issues.
        logger.debug("Calling _parse_and_validate_snippet.")
        validated_tree = _parse_and_validate_snippet(snippet_string)

        # Step 2: Execute the validated AST
        # This can raise ValueError for runtime issues within the snippet.
        logger.debug("Calling _execute_validated_snippet.")
        result = _execute_validated_snippet(validated_tree)
        
        logger.info(f"Snippet executed successfully. Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"ValueError during snippet processing: {e}")
        raise
    except TypeError as e: # Should be caught by the initial check, but good for completeness
        logger.error(f"TypeError during snippet processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected exception during snippet processing: {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred: {e}")


# Example Usage (optional, for testing):
if __name__ == '__main__':
    # Basic logging configuration for demonstration
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Starting example snippet executions.")
    print("Running examples (check console for detailed logs):")

    snippets = [
        ("1 + 1", "Simple arithmetic"),
        ("x = 10\ny = 20\nx + y", "Multi-line with result"),
        ("print('Hello from snippet!')\nx = 5\nx*x", "Print and expression"),
        ("import math\nmath.sqrt(16)", "Import math and use"),
        ("len([1, 2, 3])", "Using allowed builtin"),
        ("a = [1,2]\na.append(3)\na", "List manipulation"),
        ("def foo(x): return x*x\nfoo(3)", "Function definition and call"),
    ]

    print("\n--- Allowed Snippets ---")
    for snippet, desc in snippets:
        try:
            print(f"\nExecuting: {desc}\n{snippet}")
            result = run_python_snippet(snippet)
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")
        except TypeError as e:
            print(f"Type Error: {e}")

    error_snippets = [
        ("open('file.txt', 'w')", "Attempt to use 'open'"),
        ("import os\nos.system('echo unsafe')", "Attempt to import 'os'"),
        ("eval('1+1')", "Attempt to use 'eval'"),
        ("__import__('os').system('echo unsafe')", "Attempt to use __import__"),
        ("x = lambda: 0\nx.__class__", "Access forbidden attribute __class__"),
        ("class MyClass:\n  def __init__(self):\n    self._secret = 1\nm = MyClass()\nm._secret", "Access protected attribute _secret"),
        ("exec('print(\"unsafe\")')", "Using exec statement"),
        ("import sys", "Importing disallowed module sys"),
        ("math.cos(0); open('secrets.txt')", "Benign then malicious"),
        ("x = 1\n y = x.bit_length() \n y", "Calling an allowed method on int"), # bit_length is fine
        ("x = 1\n y = x.__add__(2) \n y", "Calling an allowed dunder method"),
        ("x = 1\n y = x.__dict__ \n y", "Accessing a disallowed dunder attribute"), # Changed from "Calling"
    ]

    print("\n--- Forbidden Snippets ---")
    for snippet, desc in error_snippets:
        try:
            print(f"\nExecuting: {desc}\n{snippet}")
            result = run_python_snippet(snippet)
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")
        except TypeError as e:
            print(f"Type Error: {e}")
            
    # Test for invalid syntax
    print("\n--- Invalid Syntax Snippet ---")
    try:
        run_python_snippet("1 +")
    except ValueError as e:
        print(f"Error: {e}")

    # Test for TypeError
    print("\n--- TypeError Snippet ---")
    try:
        run_python_snippet(123)
    except TypeError as e:
        print(f"Error: {e}") # This will be logged by run_python_snippet as well
    except ValueError as e: # Should not happen for this test
        print(f"Unexpected ValueError: {e}")
    
    logger.info("Finished example snippet executions.")
