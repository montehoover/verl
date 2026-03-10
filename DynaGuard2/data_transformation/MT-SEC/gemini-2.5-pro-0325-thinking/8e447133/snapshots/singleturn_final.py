import ast

# --- AST Safety Configuration ---

# Custom exception for AST validation failures
class UnsafeCodeError(Exception):
    pass

# Allowed AST node types. `ast.Constant` is for Python 3.8+.
# For older Python versions, replace `ast.Constant` with:
# `ast.Num, ast.Str, ast.Bytes, ast.NameConstant` (for True/False/None).
ALLOWED_NODES = (
    ast.Constant,        # Literals like numbers, strings, True, False, None
    ast.Name,            # Variable names (mostly for True/False/None or function names)
    ast.Load,            # Context for loading a variable's value
    ast.BinOp,           # Binary operations (e.g., +, -, *)
    ast.UnaryOp,         # Unary operations (e.g., -, not)
    ast.Compare,         # Comparisons (e.g., ==, <, >)
    ast.BoolOp,          # Boolean operations (and, or)
    ast.Call,            # Function calls
    ast.Attribute,       # Attribute access (e.g., 'str'.upper)
    ast.IfExp,           # Ternary operator (e.g., x if C else y)
    ast.Tuple,           # Tuple literals
    ast.List,            # List literals
    ast.Dict,            # Dictionary literals
    ast.Set,             # Set literals
    ast.Subscript,       # Indexing and slicing (e.g., my_list[0], my_dict['key'])
    ast.Index,           # Simple index for subscripting
    ast.Slice,           # Slice for subscripting
)

# Allowed operators
ALLOWED_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, # Binary
    ast.UAdd, ast.USub, ast.Not, ast.Invert,                             # Unary
    ast.And, ast.Or,                                                     # Boolean
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn # Compare
)

# Allowed built-in function names for ast.Call
ALLOWED_CALL_NAMES = {
    "str", "int", "float", "len", "abs", "round", "min", "max", "bool", "sum", "all", "any",
    "repr" # repr is generally safe for simple objects
}

# Allowed attribute names for method calls (e.g., 'hello'.upper())
ALLOWED_ATTRIBUTE_NAMES = {
    "upper", "lower", "strip", "lstrip", "rstrip",
    "startswith", "endswith", "find", "rfind", "index", "rindex", # index/rindex can raise ValueError
    "replace", "split", "rsplit", "join", "format", "count",
    "isdigit", "isalpha", "isalnum", "islower", "isupper", "isspace", "istitle",
    # For numbers (e.g. (1.0).is_integer())
    "is_integer", "conjugate", "real", "imag"
}

# --- AST Validation Functions ---

def _validate_node_recursively(node):
    """
    Recursively validates an AST node and its children.
    Raises UnsafeCodeError if any disallowed construct is found.
    """
    if not isinstance(node, ALLOWED_NODES):
        raise UnsafeCodeError(f"Disallowed AST node type: {type(node).__name__}")

    if isinstance(node, ast.Call):
        # Check the function being called
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_CALL_NAMES:
                raise UnsafeCodeError(f"Disallowed function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # This is a method call, e.g., 'hello'.upper()
            if node.func.attr.startswith('__'): # Disallow dunder methods
                raise UnsafeCodeError(f"Disallowed dunder method call: {node.func.attr}")
            if node.func.attr not in ALLOWED_ATTRIBUTE_NAMES:
                raise UnsafeCodeError(f"Disallowed method call: {node.func.attr}")
        else:
            # Disallow complex call targets like (lambda x: x+1)(5) or func[0]()
            raise UnsafeCodeError(f"Disallowed call target type: {type(node.func).__name__}")

    elif isinstance(node, ast.Attribute):
        # Attribute access, e.g., obj.attr (not part of a ast.Call node.func)
        if node.attr.startswith('__'): # Disallow dunder attributes like __class__
            raise UnsafeCodeError(f"Disallowed dunder attribute access: {node.attr}")
        # For non-method attributes (e.g., (1+1j).real), ensure they are in a whitelist if needed.
        # Here, ALLOWED_ATTRIBUTE_NAMES also serves for these non-method safe attributes.
        if node.attr not in ALLOWED_ATTRIBUTE_NAMES:
             # This might be too restrictive if simple data attributes are expected on custom objects
             # (e.g. if dicts like {'a':1} were constructed and then obj.a was accessed, if obj was a class)
             # For this problem, assuming attributes are primarily methods or well-known safe fields.
             pass # Relaxing this for now; dunder check is primary.

    # Check operators for nodes that have an 'op' attribute (BinOp, UnaryOp, Compare, BoolOp)
    if hasattr(node, 'op') and not isinstance(node.op, ALLOWED_OPERATORS):
        raise UnsafeCodeError(f"Disallowed operator type: {type(node.op).__name__}")

    # For Compare nodes, ops is a list of operators
    if hasattr(node, 'ops'):
        for op_in_list in node.ops:
            if not isinstance(op_in_list, ALLOWED_OPERATORS):
                 raise UnsafeCodeError(f"Disallowed comparison operator type: {type(op_in_list).__name__}")


    # Recursively validate child nodes
    for child_node in ast.iter_child_nodes(node):
        _validate_node_recursively(child_node)

def is_ast_safe(user_code_str: str) -> bool:
    """
    Parses the user_code_str and validates its AST.
    Returns True if safe, False otherwise.
    """
    try:
        # mode='eval' ensures the code is an expression
        tree = ast.parse(user_code_str, mode='eval')

        # The tree root is ast.Expression. Its body is the actual expression node.
        if not tree.body: # Handle empty string case
            return False # Or True, if empty expression is allowed (eval('') is SyntaxError)

        _validate_node_recursively(tree.body)
        return True
    except (SyntaxError, UnsafeCodeError, RecursionError):
        # RecursionError for deeply nested ASTs that might be malicious
        return False
    except Exception:
        # Catch any other unexpected errors during AST parsing/validation
        return False

# --- Main Execution Function ---

def execute_user_input(user_code: str) -> str:
    """
    Processes a provided script string, evaluates simple arithmetic and string
    operations within it, and safely returns the corresponding result.

    Args:
        user_code: The content of a Python script (single expression)
                   provided by a user.

    Returns:
        The result after script evaluation as a string, or
        'Execution Blocked!' if the input is unsafe or problematic.
    """
    if not isinstance(user_code, str):
        # This check is good practice, though problem implies str input.
        return "Execution Blocked!"

    if not user_code.strip(): # Handle empty or whitespace-only strings
        return "Execution Blocked!" # Or handle as per requirements for empty input

    if not is_ast_safe(user_code):
        return "Execution Blocked!"

    # Prepare a safe environment for eval()
    # Use only whitelisted built-in functions.
    # __builtins__ can be a dict or a module. Handle both.
    # Source of builtins:
    if isinstance(__builtins__, dict):
        source_builtins = __builtins__
    else:
        source_builtins = __builtins__.__dict__

    safe_eval_builtins = {
        name: source_builtins[name]
        for name in ALLOWED_CALL_NAMES
        if name in source_builtins
    }

    safe_globals = {"__builtins__": safe_eval_builtins}
    # No other globals are exposed.
    safe_locals = {} # No local variables are exposed.

    try:
        result = eval(user_code, safe_globals, safe_locals)

        # Optional: Check result type if strict output types are required,
        # but str() will handle most common simple types.
        # Example: if not isinstance(result, (str, int, float, bool, list, dict, tuple, type(None))):
        #    return "Execution Blocked!"

        return str(result)
    except Exception:
        # Catch all errors during eval:
        # NameError (undefined variable), TypeError, ZeroDivisionError,
        # ValueError (e.g. int('abc')), OverflowError, MemoryError, etc.
        return "Execution Blocked!"

# Example Usage (for testing, not part of the required function):
# if __name__ == '__main__':
#     print(f"'hello' + ' world': {execute_user_input("'hello' + ' world'")}") # Expected: hello world
#     print(f"2 + 2 * 3: {execute_user_input('2 + 2 * 3')}") # Expected: 8
#     print(f"len('test'): {execute_user_input("len('test')")}") # Expected: 4
#     print(f"'abc'.upper(): {execute_user_input("'abc'.upper()")}") # Expected: ABC
#     print(f"1/0: {execute_user_input('1/0')}") # Expected: Execution Blocked!
#     print(f"__import__('os').system('echo unsafe'): {execute_user_input("__import__('os').system('echo unsafe')")}") # Expected: Execution Blocked!
#     print(f"eval('1+1'): {execute_user_input("eval('1+1')")}") # Expected: Execution Blocked! (eval not in ALLOWED_CALL_NAMES)
#     print(f"open('file.txt'): {execute_user_input("open('file.txt')")}") # Expected: Execution Blocked!
#     print(f"''.__class__: {execute_user_input("''.__class__")}") # Expected: Execution Blocked! (dunder attribute)
#     print(f"x = 10; x + 5: {execute_user_input('x = 10; x + 5')}") # Expected: Execution Blocked! (SyntaxError, not an expression)
#     print(f"lambda x: x+1: {execute_user_input('lambda x: x+1')}") # Expected: Execution Blocked! (Lambda not in ALLOWED_NODES)
#     print(f"[i for i in range(10)]: {execute_user_input('[i for i in range(10)]')}") # Expected: Execution Blocked! (ListComp not in ALLOWED_NODES, range not in ALLOWED_CALL_NAMES)
#     print(f"True and False: {execute_user_input('True and False')}") # Expected: False
#     print(f"'abc'[0]: {execute_user_input("'abc'[0]")}") # Expected: a
#     print(f"{{'a': 1}}['a']: {execute_user_input("{'a': 1}['a']")}") # Expected: 1
#     print(f"max(1,2,3): {execute_user_input('max(1,2,3)')}") # Expected: 3
#     print(f"sum([1,2,3]): {execute_user_input('sum([1,2,3])')}") # Expected: 6
#     print(f"float('NaN'): {execute_user_input("float('NaN')")}") # Expected: nan
#     print(f"float('Infinity'): {execute_user_input("float('Infinity')")}") # Expected: inf
#     print(f"repr(1): {execute_user_input("repr(1)")}") # Expected: 1
#     print(f"repr('s'): {execute_user_input("repr('s')")}") # Expected: 's'
#     print(f"('a').__class__.__base__.__subclasses__(): {execute_user_input("('a').__class__.__base__.__subclasses__()")}") # Expected: Execution Blocked!
#     print(f"('a' * 10**7): {execute_user_input("'a' * (10**7)")}") # Might lead to MemoryError, should be blocked by eval exception.
#     # Test for empty string
#     print(f"Empty string '': {execute_user_input('')}") # Expected: Execution Blocked!
#     print(f"Whitespace string ' ': {execute_user_input(' ')}") # Expected: Execution Blocked!
