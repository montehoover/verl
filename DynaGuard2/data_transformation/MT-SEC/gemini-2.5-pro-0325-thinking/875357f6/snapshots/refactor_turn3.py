import ast
import io
import contextlib
import types # For CodeType type hint
import logging

# Configure logger
logger = logging.getLogger(__name__)
# Example basic configuration:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# This basicConfig should ideally be called once at the application entry point, not in a library module.
# For this exercise, we'll assume it's configured elsewhere or rely on default settings.


class SafeNodeVisitor(ast.NodeVisitor):
    """
    Traverses an AST to ensure it doesn't contain forbidden operations.
    Raises ValueError if a forbidden operation is found.
    """
    # eval, exec, open, compile, input, __import__ are inherently dangerous
    # getattr, setattr, delattr can be used to bypass restrictions
    # globals, locals, vars can expose too much scope information if not controlled
    FORBIDDEN_CALLS = {
        'eval', 'exec', 'open', 'compile', 'input', '__import__',
        'getattr', 'setattr', 'delattr',
        'globals', 'locals', 'vars',
    }

    # __builtins__ can be used to access dangerous functions if not shadowed
    FORBIDDEN_NAMES_ACCESS = {'__builtins__'} 

    # Attributes commonly used in exploits or to access sensitive internals
    FORBIDDEN_ATTRIBUTES = {
        '__builtins__', '__class__', '__closure__', '__code__', '__dict__',
        '__file__', '__func__', '__globals__', '__import__', '__mro__',
        '__bases__', '__name__', '__path__', '__self__', '__subclasses__',
        # Frame object attributes
        'f_back', 'f_builtins', 'f_code', 'f_globals', 'f_locals', 'f_trace',
        # Code object attributes
        'co_argcount', 'co_cellvars', 'co_code', 'co_consts', 'co_filename',
        'co_firstlineno', 'co_flags', 'co_freevars', 'co_kwonlyargcount',
        'co_lnotab', 'co_name', 'co_names', 'co_nlocals', 'co_posonlyargcount',
        'co_stacksize', 'co_varnames',
        # Generator attributes
        'gi_code', 'gi_frame', 'gi_running', 'gi_yieldfrom',
        # OS interaction methods (if an object having them is obtained)
        'system', 'popen', 'listdir', 'remove', 'unlink', 'rmdir', 'chmod', 'chown',
        'read', 'write', 'send', 'connect', # common I/O method names
    }

    def visit_Import(self, node):
        raise ValueError("Import statements (import ...) are forbidden.")
    
    def visit_ImportFrom(self, node):
        raise ValueError("Import statements (from ... import ...) are forbidden.")

    def visit_Call(self, node):
        # Check if the function being called is an ast.Name (e.g., func())
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                raise ValueError(f"Call to forbidden function '{node.func.id}' is not allowed.")
        # Check if the function being called is an ast.Attribute (e.g., obj.func())
        # This catches calls like `some_object.eval(...)` if 'eval' is in FORBIDDEN_CALLS.
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_CALLS:
                 raise ValueError(f"Call to forbidden method name '{node.func.attr}' is not allowed.")
        
        # Recursively visit children (arguments, keywords, etc.)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # node.value is the object, node.attr is the attribute name (string)
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            raise ValueError(f"Access to forbidden attribute '{node.attr}' is not allowed.")
        self.generic_visit(node) # Visit the .value part and any other children

    def visit_Name(self, node):
        # Prevent direct loading of names like '__builtins__'
        # e.g. x = __builtins__ (where ctx is ast.Load)
        if node.id in self.FORBIDDEN_NAMES_ACCESS and isinstance(node.ctx, ast.Load):
            raise ValueError(f"Direct access to forbidden name '{node.id}' is not allowed.")
        self.generic_visit(node)
    
    # Disallow features that complicate security or are often restricted
    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Async function definitions (async def) are forbidden.")

    def visit_Await(self, node):
        raise ValueError("Await expressions are forbidden.")

    def visit_AsyncFor(self, node):
        raise ValueError("Async for loops are forbidden.")

    def visit_AsyncWith(self, node):
        raise ValueError("Async with statements are forbidden.")

    # Exec statement (Python 2 syntax, ast.parse might still create this node)
    # In Python 3, exec() is a function, handled by visit_Call if 'exec' is in FORBIDDEN_CALLS.
    def visit_Exec(self, node):
        raise ValueError("The 'exec' statement (Python 2 style) is forbidden.")


def _validate_script_ast(tree: ast.AST):
    """
    Validates the given AST using SafeNodeVisitor.

    Args:
        tree: The Abstract Syntax Tree of the script.

    Raises:
        ValueError: If the script uses forbidden operations.
    """
    validator = SafeNodeVisitor()
    validator.visit(tree)


def _execute_validated_script(code_object: types.CodeType, safe_builtins: dict):
    """
    Executes a compiled code object in a restricted environment.

    Args:
        code_object: The compiled code to execute.
        safe_builtins: A dictionary of whitelisted built-in functions and types.

    Returns:
        The standard output produced by the script as a string, if any.
        Returns None if the script produces no output or if a runtime
        error occurs during its execution.
    """
    restricted_globals = {"__builtins__": safe_builtins}
    restricted_locals = {}
    output_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code_object, restricted_globals, restricted_locals)
    except Exception as e:
        # Runtime errors in the user's script
        logger.error(f"Runtime error during script execution: {e}", exc_info=True)
        return None

    final_output = output_capture.getvalue()
    return final_output if final_output else None


def execute_user_script(submitted_script: str):
    """
    Safely executes a Python script provided by the user by parsing it,
    validating its AST against a list of forbidden operations, and then
    executing it in a restricted environment.

    Args:
        submitted_script: The Python script string to execute.

    Returns:
        The standard output produced by the script as a string, if any.
        Returns None if the script produces no output or if a runtime
        error occurs during its execution (after passing validation).

    Raises:
        ValueError: If the script contains syntax errors or uses forbidden
                    operations identified during AST validation.
    """
    try:
        # Parse the script into an Abstract Syntax Tree (AST)
        # filename='<user_script>' is good practice for error messages from ast/compile
        script_preview = submitted_script[:200] + ('...' if len(submitted_script) > 200 else '')
        logger.info(f"Attempting to execute user script (preview): {script_preview}")
        tree = ast.parse(submitted_script, filename='<user_script>')
    except SyntaxError as e:
        # As per prompt: "raise a ValueError if ... contains syntax errors"
        logger.error(f"Syntax error in submitted script: {e}", exc_info=True)
        raise ValueError(f"Syntax error in submitted script: {e}") from e

    # Validate the AST
    try:
        _validate_script_ast(tree)
        logger.info("Script AST validation successful.")
    except ValueError as e:
        # Re-raise ValueError from the validator (indicates forbidden action)
        logger.warning(f"Script AST validation failed: {e}", exc_info=True)
        raise

    # Define a whitelist of safe built-in functions and types for the execution scope.
    safe_builtins = {
        'print': print,
        'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'len': len,
        'str': str, 'int': int, 'float': float, 'bool': bool, 
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'range': range, 'isinstance': isinstance, 'issubclass': issubclass,
        'True': True, 'False': False, 'None': None,
        # Common exception types that user scripts might legitimately raise or catch
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError, 
        'IndexError': IndexError, 'KeyError': KeyError, 'AttributeError': AttributeError,
        'NameError': NameError, 'ZeroDivisionError': ZeroDivisionError,
        # Other relatively safe builtins that don't typically offer direct I/O or system access:
        'all': all, 'any': any, 'bin': bin, 'callable': callable, 'chr': chr, 
        'complex': complex, 'divmod': divmod, 'enumerate': enumerate, 'filter': filter, 
        'format': format, 'frozenset': frozenset, 'hash': hash, 'hex': hex, 
        'id': id, 'map': map, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow, 
        'repr': repr, 'reversed': reversed, 'slice': slice, 'sorted': sorted, 
        'type': type, 'zip': zip,
        # Note: 'dir' could be added if introspection within the sandbox is acceptable.
        # Functions like 'getattr', 'setattr', 'eval', 'open', etc., are kept out of
        # this list and are also checked by the AST visitor.
    }
    
    try:
        # Compile the validated AST into a code object.
        code_object = compile(tree, filename='<user_script>', mode='exec')
    except Exception as e: # Includes potential errors during compilation itself
        # While ast.parse catches syntax errors, compile can find other issues.
        # Treat as a form of invalid script, though prompt focuses ValueError on syntax/forbidden.
        # For robustness, if compile fails, it's akin to a script that cannot run.
        # Returning None aligns with "runtime error" leading to None.
        # Alternatively, one could raise a ValueError here too.
        # Given the prompt's strictness on ValueError for syntax/forbidden,
        # and None for runtime errors, this seems more aligned.
        logger.error(f"Error compiling script AST: {e}", exc_info=True)
        return None # Or raise ValueError(f"Error compiling script: {e}")
    
    logger.info("Script compiled successfully.")

    # Execute the compiled code and capture output
    result = _execute_validated_script(code_object, safe_builtins)

    if result is not None:
        output_preview = result[:200] + ('...' if len(result) > 200 else '')
        logger.info(f"Script executed successfully. Output (preview): {output_preview}")
    else:
        # This case covers runtime errors within _execute_validated_script (already logged there)
        # or if the script produced no output and _execute_validated_script returned None.
        logger.info("Script executed, but produced no output or encountered a runtime error (see previous logs if any).")
        
    return result
