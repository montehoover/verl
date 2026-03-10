import ast
import logging

# Configure basic logging if no handlers are configured
# In a real application, this might be configured at a higher level
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Maximum length for the input Python code string
MAX_CODE_LENGTH = 1024
# Maximum length for the string representation of the result
MAX_RESULT_LENGTH = 4096

# Whitelist of built-in function names and safe objects for the evaluation context
SAFE_BUILTINS = {
    'abs': abs, 'round': round, 'len': len,
    'str': str, 'int': int, 'float': float, 'bool': bool,
    'min': min, 'max': max, 'divmod': divmod,
    'True': True, 'False': False, 'None': None,
    # Allowing list and tuple constructors for use with e.g. string join
    'list': list, 'tuple': tuple,
}

# Whitelist of function names allowed in ast.Call nodes
ALLOWED_CALL_NAMES = {
    'abs', 'round', 'len', 'str', 'int', 'float', 'bool', 'min', 'max', 'divmod',
    'list', 'tuple', # if constructing lists/tuples like list([1,2])
}

# Whitelist of string methods allowed in ast.Attribute nodes
ALLOWED_STRING_METHODS = {
    'upper', 'lower', 'strip', 'lstrip', 'rstrip', 'startswith', 'endswith',
    'replace', 'split', 'join', 'count', 'find',
    'isdigit', 'isalpha', 'isalnum', 'isspace', 'istitle', 'islower', 'isupper',
    'capitalize', 'title',
}

class SecureAstValidator(ast.NodeVisitor):
    def __init__(self, allowed_names=None):
        super().__init__()
        self.is_code_safe = True
        self.allowed_names = allowed_names if allowed_names is not None else set(SAFE_BUILTINS.keys())

        self.allowed_nodes = {
            ast.Expression, ast.Constant, ast.Name,
            ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
            ast.Call, ast.Attribute,
            ast.List, ast.Tuple,  # For literals like ['a', 'b'] in "'_'.join(['a', 'b'])"
            ast.Subscript,        # For s[0] or s[1:3]
            ast.Slice,            # For the s[1:3] part (represents the slice object)
        }
        self.allowed_bin_ops = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        }
        self.allowed_unary_ops = {ast.UAdd, ast.USub, ast.Not}
        self.allowed_comp_ops = {
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
        }
        self.allowed_bool_ops = {ast.And, ast.Or}

    def visit(self, node):
        if not self.is_code_safe:
            return

        node_type = type(node)
        if node_type not in self.allowed_nodes:
            self.is_code_safe = False
            return

        if isinstance(node, ast.Name):
            # Allow names if they are in SAFE_BUILTINS (for values like True) or ALLOWED_CALL_NAMES (for functions)
            if node.id not in self.allowed_names and node.id not in ALLOWED_CALL_NAMES:
                self.is_code_safe = False
                return
        elif isinstance(node, ast.BinOp) and type(node.op) not in self.allowed_bin_ops:
            self.is_code_safe = False
            return
        elif isinstance(node, ast.UnaryOp) and type(node.op) not in self.allowed_unary_ops:
            self.is_code_safe = False
            return
        elif isinstance(node, ast.Compare):
            for op_type in node.ops:
                if type(op_type) not in self.allowed_comp_ops:
                    self.is_code_safe = False
                    return
        elif isinstance(node, ast.BoolOp) and type(node.op) not in self.allowed_bool_ops:
            self.is_code_safe = False
            return
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_CALL_NAMES:
                    self.is_code_safe = False
                    return
            elif isinstance(node.func, ast.Attribute):
                # Safety of the attribute itself (e.g., whitelisted method) is checked by visit_Attribute
                pass
            else: # Calling something other than a Name or Attribute
                self.is_code_safe = False
                return
        elif isinstance(node, ast.Attribute):
            # Allow attribute access if the attribute name is a whitelisted string method.
            # The object (node.value) on which the attribute is accessed will be visited by generic_visit.
            if node.attr not in ALLOWED_STRING_METHODS:
                self.is_code_safe = False
                return

        super().generic_visit(node) # Recursively visit children

    def validate(self, code_string: str) -> bool:
        try:
            tree = ast.parse(code_string, mode='eval') # mode='eval' for a single expression
            self.is_code_safe = True # Reset for potential reuse of validator instance
            self.visit(tree)
            return self.is_code_safe
        except (SyntaxError, ValueError, TypeError, RecursionError): # Catch parsing errors or too complex ASTs
            return False


def _is_code_safe(python_code: str) -> bool:
    """
    Checks if the Python code string is safe for evaluation.
    Performs initial length check and AST validation.
    """
    if not isinstance(python_code, str) or len(python_code) > MAX_CODE_LENGTH:
        return False

    validator = SecureAstValidator()
    if not validator.validate(python_code):
        return False
    return True


def _execute_safely(python_code: str) -> str:
    """
    Executes the validated Python code string and returns the result.
    Raises exceptions if compilation, execution, or result validation fails.
    """
    # Compile the source code. mode='eval' as we expect a single expression.
    # SyntaxError during compile should ideally be caught by ast.parse in _is_code_safe
    code_object = compile(python_code, '<user_code>', 'eval')

    # Verify that no unexpected names are present in the compiled code object.
    for name in code_object.co_names:
            if name not in SAFE_BUILTINS and name not in ALLOWED_CALL_NAMES and name not in ALLOWED_STRING_METHODS:
                raise ValueError("Disallowed name found in code object")

    # Evaluate the compiled code in a restricted environment.
    safe_globals = {'__builtins__': {}}
    safe_globals.update(SAFE_BUILTINS)
    
    result = eval(code_object, safe_globals, {})

    # Ensure the result is of a simple, expected type before stringifying.
    if isinstance(result, (list, tuple)):
        for item in result:
            if not isinstance(item, (str, int, float, bool, type(None))):
                raise TypeError("Disallowed type found in list/tuple result")
    elif not isinstance(result, (str, int, float, bool, type(None))):
        raise TypeError("Disallowed result type")

    result_str = str(result)
    if len(result_str) > MAX_RESULT_LENGTH:
        raise ValueError("Result string too long")

    return result_str


def run_user_code(python_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script string.
    Supported operations are limited to basic arithmetic and string manipulations.

    Args:
        python_code: A user-supplied Python script (single expression).

    Returns:
        The result of evaluating the script as a string, or 'Execution Blocked!'
        if the script is found to be unsafe or an error occurs.
    """
    logger.info(f"Attempting to evaluate user code: {python_code[:200]}{'...' if len(python_code) > 200 else ''}")

    if not _is_code_safe(python_code):
        logger.warning(f"Execution blocked for unsafe code: {python_code[:200]}{'...' if len(python_code) > 200 else ''}")
        return "Execution Blocked!"

    try:
        result = _execute_safely(python_code)
        logger.info(f"Successfully evaluated code. Result: {result[:200]}{'...' if len(result) > 200 else ''}")
        return result
    except Exception as e:
        logger.error(f"Exception during execution of code: {python_code[:200]}{'...' if len(python_code) > 200 else ''}. Error: {e}", exc_info=True)
        # Catches exceptions from _execute_safely (e.g., compilation, disallowed names,
        # eval errors, disallowed result types, result too long)
        return "Execution Blocked!"
