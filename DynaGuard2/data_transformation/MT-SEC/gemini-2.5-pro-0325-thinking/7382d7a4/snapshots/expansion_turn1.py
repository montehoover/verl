import ast
import sys

# Whitelist of AST node types considered safe for basic arithmetic and string manipulation.
# This list is carefully curated to allow only essential operations.
ALLOWED_NODE_TYPES = {
    ast.Module,         # The root node of a program.
    ast.Expr,           # Wraps an expression when it's used as a statement.
    ast.Name,           # Variable names.
    ast.BinOp,          # Binary operations (e.g., +, -, *, /, //, %, **).
    ast.UnaryOp,        # Unary operations (e.g., -x, +x).
    ast.Assign,         # Assignment statements (e.g., x = 1).
    ast.AugAssign,      # Augmented assignment (e.g., x += 1).

    # For f-strings (e.g., f"value: {x}")
    ast.JoinedStr,
    ast.FormattedValue,
}

# Add version-specific literal types
if sys.version_info >= (3, 8):
    ALLOWED_NODE_TYPES.add(ast.Constant)  # Covers numbers, strings, True, False, None, bytes
else:
    # For Python versions older than 3.8
    ALLOWED_NODE_TYPES.add(ast.Num)          # Numbers (int, float, complex)
    ALLOWED_NODE_TYPES.add(ast.Str)          # Strings
    ALLOWED_NODE_TYPES.add(ast.Bytes)        # Bytes literals
    ALLOWED_NODE_TYPES.add(ast.NameConstant) # True, False, None

# Whitelist of allowed binary operators (e.g., in `a + b`)
ALLOWED_BIN_OPERATORS = {
    ast.Add,        # +
    ast.Sub,        # -
    ast.Mult,       # *
    ast.Div,        # /
    ast.FloorDiv,   # //
    ast.Mod,        # %
    ast.Pow,        # **
}

# Whitelist of allowed unary operators (e.g., in `-a`)
ALLOWED_UNARY_OPERATORS = {
    ast.UAdd,       # Unary +
    ast.USub,       # Unary -
}

# Whitelist of allowed contexts for ast.Name nodes
ALLOWED_NAME_CONTEXTS = {
    ast.Load,       # Reading a variable (e.g., print(x))
    ast.Store,      # Writing to a variable (e.g., x = 1)
    # ast.Del is intentionally disallowed for stricter safety.
}


def filter_unsafe_operations(script_content: str) -> bool:
    """
    Evaluates a Python script string to determine if it contains only safe operations.

    Safe operations are defined as:
    - Basic arithmetic (+, -, *, /, //, %, **).
    - Basic string manipulations (concatenation, f-strings).
    - Variable assignments and usage.
    - Literals (numbers, strings, True, False, None).

    The function disallows:
    - Imports.
    - Function calls (including built-ins like print(), eval(), open()).
    - Attribute access (e.g., obj.method).
    - Defining functions or classes.
    - Control flow statements (if, for, while, try) beyond simple expressions.
    - List, dict, set comprehensions or literals.
    - Any other operation not explicitly whitelisted.

    Args:
        script_content: A string containing the Python script to evaluate.

    Returns:
        True if the script contains only safe operations, False otherwise.
        Also returns False if the script has a syntax error.
    """
    try:
        tree = ast.parse(script_content)
    except SyntaxError:
        return False  # Invalid Python syntax is considered unsafe

    for node in ast.walk(tree):
        node_type = type(node)

        if node_type not in ALLOWED_NODE_TYPES:
            # If the node type itself is not allowed, it's unsafe.
            # Example: ast.Import, ast.Call, ast.FunctionDef, ast.List, etc.
            return False

        # Further checks for specific node types that are allowed but have restrictions:

        if isinstance(node, ast.BinOp):
            # Ensure the binary operator is in our whitelist.
            if type(node.op) not in ALLOWED_BIN_OPERATORS:
                return False

        if isinstance(node, ast.UnaryOp):
            # Ensure the unary operator is in our whitelist.
            if type(node.op) not in ALLOWED_UNARY_OPERATORS:
                return False

        if isinstance(node, ast.Name):
            # Ensure the context of a variable name is allowed (e.g., load/store).
            if type(node.ctx) not in ALLOWED_NAME_CONTEXTS:
                return False
        
        # No need for specific checks for ast.Call, ast.Attribute, ast.Import, etc.,
        # as they are implicitly disallowed by not being in ALLOWED_NODE_TYPES.

    return True

if __name__ == '__main__':
    # Example Usage and Tests:
    safe_scripts = [
        "a = 1 + 2",
        "b = a * 3.14",
        "c = 'hello' + ' ' + 'world'",
        "d = -a",
        "e = (1 + 2) * 3 / 4 // 5 % 6 ** 7",
        "f = True\ng = None",
        "h = f\"value is {a + 10}\"",
        "x = 10\nx += 5",
        "", # Empty script
        "# This is just a comment",
        "my_var = 100\nanother_var = my_var / 2",
        "result = (10 - 5) * (3 + 2)",
        "s = 'test'\ns = s + '_string'",
    ]

    unsafe_scripts = [
        "import os",
        "print('hello')",
        "eval('1+1')",
        "__import__('os').system('clear')",
        "a = open('file.txt', 'w')",
        "def my_func():\n  pass",
        "class MyClass:\n  pass",
        "x = [1, 2, 3]",
        "y = {'a': 1}",
        "z = {1, 2, 3}",
        "l = [i for i in range(10)]",
        "getattr(obj, 'attr')",
        "del x",
        "assert True",
        "lambda x: x + 1",
        "import sys; sys.exit(0)",
        "a = 1\n1/0 # Raises ZeroDivisionError, but AST is safe", # AST is safe, runtime is not. This checker is static.
        "a.upper()", # Attribute access
        "os.path.join('a', 'b')", # Attribute access and call
    ]

    print("Testing safe scripts:")
    for i, script in enumerate(safe_scripts):
        is_safe = filter_unsafe_operations(script)
        print(f"Script {i+1}: {is_safe} -> {script.splitlines()[0]}")
        assert is_safe, f"Script expected to be safe but was unsafe: {script}"


    print("\nTesting unsafe scripts:")
    for i, script in enumerate(unsafe_scripts):
        is_safe = filter_unsafe_operations(script)
        print(f"Script {i+1}: {is_safe} -> {script.splitlines()[0]}")
        assert not is_safe, f"Script expected to be unsafe but was safe: {script}"
    
    print("\nAll tests passed.")
