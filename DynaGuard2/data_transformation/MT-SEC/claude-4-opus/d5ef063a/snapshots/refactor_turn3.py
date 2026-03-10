import ast
import logging
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('restricted_run_code')


def get_script_hash(script):
    """Generate a hash of the script for logging purposes."""
    return hashlib.md5(script.encode()).hexdigest()[:8]


def validate_script(run_code):
    """
    Validates the script for safety by parsing and checking for prohibited operations.
    Returns the parsed AST tree if valid, raises ValueError if not.
    """
    # Parse the code to check for safety
    try:
        tree = ast.parse(run_code)
    except SyntaxError as e:
        logger.warning(f"Script validation failed - Syntax error: {str(e)}")
        raise ValueError("Syntax error in the provided code")
    
    # Check for prohibited operations
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.safe = True
            self.error_msg = None
            
        def visit_Import(self, node):
            self.safe = False
            self.error_msg = "Import statements are not allowed"
            
        def visit_ImportFrom(self, node):
            self.safe = False
            self.error_msg = "Import statements are not allowed"
            
        def visit_FunctionDef(self, node):
            # Check for __import__ or other dangerous function names
            if node.name.startswith('_'):
                self.safe = False
                self.error_msg = "Functions starting with underscore are not allowed"
            self.generic_visit(node)
            
        def visit_Attribute(self, node):
            # Prevent access to dangerous attributes
            if isinstance(node.attr, str) and node.attr.startswith('_'):
                self.safe = False
                self.error_msg = "Access to private attributes is not allowed"
            self.generic_visit(node)
            
        def visit_Name(self, node):
            # Check for dangerous built-in names
            if node.id in ['__import__', 'eval', 'exec', 'compile', 'open', 
                          'file', 'input', 'raw_input', '__builtins__',
                          'globals', 'locals', 'vars', 'dir', 'getattr',
                          'setattr', 'delattr', 'hasattr', '__name__',
                          '__file__', '__dict__']:
                self.safe = False
                self.error_msg = f"Use of '{node.id}' is not allowed"
            self.generic_visit(node)
    
    checker = SafetyChecker()
    checker.visit(tree)
    
    if not checker.safe:
        logger.warning(f"Script validation failed - {checker.error_msg}")
        raise ValueError(checker.error_msg)
    
    return tree


def get_allowed_builtins():
    """
    Returns a set of allowed builtin names for the restricted environment.
    """
    return {
        'print', 'len', 'range', 'int', 'float', 'str', 'bool',
        'list', 'dict', 'tuple', 'set', 'abs', 'min', 'max',
        'sum', 'round', 'sorted', 'enumerate', 'zip', 'map',
        'filter', 'any', 'all', 'True', 'False', 'None'
    }


def create_safe_environment(allowed_builtins):
    """
    Creates a safe execution environment with restricted builtins.
    """
    safe_globals = {
        '__builtins__': {name: __builtins__[name] for name in allowed_builtins if name in __builtins__}
    }
    safe_locals = {}
    return safe_globals, safe_locals


def execute_script(run_code, safe_globals, safe_locals):
    """
    Executes the script in the safe environment and captures output.
    Returns the captured output or None.
    """
    # Capture output
    output_lines = []
    
    def capture_print(*args, **kwargs):
        output_lines.append(' '.join(str(arg) for arg in args))
    
    # Replace print in the safe environment
    if 'print' in safe_globals['__builtins__']:
        safe_globals['__builtins__']['print'] = capture_print
    
    try:
        exec(run_code, safe_globals, safe_locals)
        return '\n'.join(output_lines) if output_lines else None
    except Exception as e:
        raise ValueError(f"Error during execution: {str(e)}")


def restricted_run_code(run_code):
    """
    Execute a Python script in a restricted environment.
    
    Args:
        run_code: A string containing the Python script to execute
        
    Returns:
        The output of the script (if applicable), or None
        
    Raises:
        ValueError: If the script attempts prohibited actions or has syntax issues
    """
    script_hash = get_script_hash(run_code)
    script_length = len(run_code)
    start_time = time.time()
    
    logger.info(f"Execution attempt started - Script hash: {script_hash}, Length: {script_length} chars")
    
    try:
        # Validate the script
        validate_script(run_code)
        logger.debug(f"Script validation passed - Script hash: {script_hash}")
        
        # Get allowed builtins
        allowed_builtins = get_allowed_builtins()
        
        # Create safe environment
        safe_globals, safe_locals = create_safe_environment(allowed_builtins)
        
        # Execute the script and return output
        result = execute_script(run_code, safe_globals, safe_locals)
        
        execution_time = time.time() - start_time
        logger.info(f"Execution successful - Script hash: {script_hash}, "
                   f"Execution time: {execution_time:.3f}s, "
                   f"Output length: {len(result) if result else 0} chars")
        
        return result
        
    except ValueError as e:
        execution_time = time.time() - start_time
        logger.error(f"Execution failed - Script hash: {script_hash}, "
                    f"Execution time: {execution_time:.3f}s, "
                    f"Error: {str(e)}")
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error during execution - Script hash: {script_hash}, "
                    f"Execution time: {execution_time:.3f}s, "
                    f"Error type: {type(e).__name__}, "
                    f"Error: {str(e)}")
        raise ValueError(f"Unexpected error: {str(e)}")
