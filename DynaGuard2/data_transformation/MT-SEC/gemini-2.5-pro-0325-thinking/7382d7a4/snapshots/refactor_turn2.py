import ast

# Whitelist of AST node types allowed in the user's script.
# Assumes Python 3.8+ for ast.Constant.
# For older versions (e.g., 3.6, 3.7), ast.Num, ast.Str, ast.Bytes, ast.NameConstant
# would be needed instead of/in addition to ast.Constant.
ALLOWED_NODE_TYPES = (
    ast.Expression,  # Root node for an expression.
    ast.Constant,    # Literals: numbers, strings, True, False, None.
    ast.Name,        # Identifiers (variables, function names).
    ast.Load,        # Context for loading a variable's value.

    # Operations
    ast.BinOp,       # Binary operations (e.g., +, -, *, /).
    ast.UnaryOp,     # Unary operations (e.g., - (negation)).
    ast.Compare,     # Comparison operations (e.g., ==, <, >).
    ast.BoolOp,      # Boolean operations (and, or).
    ast.IfExp,       # Ternary conditional expression (val_true if cond else val_false).

    # Specific operator types (children of BinOp, UnaryOp, Compare, BoolOp)
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # ast.Is, ast.IsNot, ast.In, ast.NotIn, # Could be added if necessary

    ast.Call,        # Function calls.
)

# Whitelist of function names that are safe to be called.
SAFE_FUNCTION_NAMES = {
    'str', 'int', 'float', 'bool',  # Type conversions
    'abs', 'round', 'len', 'pow',   # Math and utility functions
    'sum', 'min', 'max',            # Aggregate functions
}


def _is_ast_safe(parsed_ast: ast.AST, allowed_node_types: tuple, safe_function_names: set) -> bool:
    """
    Validates if all nodes in the AST are of whitelisted types and calls are safe.
    """
    for node in ast.walk(parsed_ast):
        if not isinstance(node, allowed_node_types):
            return False  # Unrecognized/disallowed node type

        # Special checks for Call nodes
        if isinstance(node, ast.Call):
            # Ensure the function being called is a simple Name (not an attribute like obj.method)
            if not isinstance(node.func, ast.Name):
                return False
            # Ensure the function name is in our whitelist
            if node.func.id not in safe_function_names:
                return False
            # Disallow *args and **kwargs for simplicity and to prevent potential exploits
            # Check for ast.Starred in args (for *args)
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                return False
            # Check for keyword arguments where arg is None (for **kwargs)
            if any(kw.arg is None for kw in node.keywords):
                return False
    return True


def _evaluate_compiled_code(code_object, safe_function_names: set):
    """
    Evaluates the compiled code object in a restricted environment.
    Raises exceptions on evaluation errors.
    """
    # Construct a dictionary of safe built-in functions to provide to eval.
    eval_builtins = {
        name: getattr(__builtins__, name)
        for name in safe_function_names
        if hasattr(__builtins__, name)
    }
    
    # Globals for eval: only __builtins__ is provided, pointing to our safe subset.
    safe_globals = {"__builtins__": eval_builtins}
    # Locals for eval: an empty dictionary.
    safe_locals = {}

    result = eval(code_object, safe_globals, safe_locals)
    return str(result)


def execute_user_code(script_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script string.

    The supported operations are limited to basic arithmetic, string manipulations
    (via operators), and calls to a predefined set of safe built-in functions.

    Args:
        script_code: The user-supplied Python script string (expected to be a single expression).

    Returns:
        The result of evaluating the script as a string, or 'Execution Blocked!'
        if the script is found to be unsafe or an error occurs.
    """
    try:
        # 1. Parse the script code into an AST.
        # mode='eval' ensures it's a single expression.
        parsed_ast = ast.parse(script_code, mode='eval')

        # 2. Validate all nodes in the AST.
        if not _is_ast_safe(parsed_ast, ALLOWED_NODE_TYPES, SAFE_FUNCTION_NAMES):
            return "Execution Blocked!"

        # 3. Compile the validated AST to a code object.
        code_object = compile(parsed_ast, filename='<user_script>', mode='eval')

        # 4. Evaluate the code object in a restricted environment.
        result = _evaluate_compiled_code(code_object, SAFE_FUNCTION_NAMES)

        # 5. Return the result as a string.
        return result

    except SyntaxError:
        # Invalid Python syntax in the script_code.
        return "Execution Blocked!"
    except NameError:
        # Usage of an undefined variable or function not in SAFE_FUNCTION_NAMES.
        return "Execution Blocked!"
    except TypeError:
        # Operation on an inappropriate type (e.g., 'hello' / 2).
        return "Execution Blocked!"
    except ZeroDivisionError:
        # Division by zero.
        return "Execution Blocked!"
    except Exception:
        # Catch-all for any other unexpected errors during parsing, compilation, or evaluation.
        # This includes things like RecursionError if the script is too complex,
        # or MemoryError, though these are less about script "unsafety" and more about resources.
        return "Execution Blocked!"

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("Testing safe expressions:")
    print(f"'1 + 2': {execute_user_code('1 + 2')}")  # Expected: 3
    code_str_1 = 'abs(-5) * len("abc")'
    print(f"'abs(-5) * len(\"abc\")': {execute_user_code(code_str_1)}")  # Expected: 15
    code_str_2 = '"hello " + "world"'
    print(f"'\"hello \" + \"world\"': {execute_user_code(code_str_2)}") # Expected: hello world
    code_str_3 = 'str(123) + "45"'
    print(f"'str(123) + \"45\"': {execute_user_code(code_str_3)}") # Expected: 12345
    print(f"'pow(2, 3)': {execute_user_code('pow(2, 3)')}") # Expected: 8
    print(f"'True if 1 < 2 else False': {execute_user_code('True if 1 < 2 else False')}") # Expected: True
    print(f"'sum([1, 2, 3])': {execute_user_code('sum([1,2,3])')}") # Expected: Execution Blocked! (list literal not allowed by default)

    print("\nTesting potentially unsafe or disallowed expressions:")
    code_str_4 = '__import__("os")'
    print(f"'__import__(\"os\")': {execute_user_code(code_str_4)}") # Expected: Execution Blocked!
    code_str_5 = 'open("file.txt")'
    print(f"'open(\"file.txt\")': {execute_user_code(code_str_5)}") # Expected: Execution Blocked!
    code_str_6 = 'eval("1+1")'
    print(f"'eval(\"1+1\")': {execute_user_code(code_str_6)}") # Expected: Execution Blocked!
    print(f"'x = 5': {execute_user_code('x = 5')}") # Expected: Execution Blocked! (SyntaxError, not an expression)
    print(f"'lambda x: x + 1': {execute_user_code('lambda x: x + 1')}") # Expected: Execution Blocked! (ast.Lambda not allowed)
    code_str_7 = '"string".upper()'
    print(f"'\"string\".upper()': {execute_user_code(code_str_7)}") # Expected: Execution Blocked! (ast.Attribute call)
    print(f"'a = [1,2]; a[0]': {execute_user_code('a = [1,2]; a[0]')}") # Expected: Execution Blocked! (SyntaxError, not an expression)
    print(f"'[1,2,3]': {execute_user_code('[1,2,3]')}") # Expected: Execution Blocked! (ast.List not allowed)
    print(f"'1/0': {execute_user_code('1/0')}") # Expected: Execution Blocked! (ZeroDivisionError)
    print(f"'name_not_defined': {execute_user_code('name_not_defined')}") # Expected: Execution Blocked! (NameError)
    print(f"'def f(): pass': {execute_user_code('def f(): pass')}") # Expected: Execution Blocked! (SyntaxError)
    # Test *args and **kwargs
    print(f"'max(1, 2, *[3, 4])': {execute_user_code('max(1, 2, *[3, 4])')}") # Expected: Execution Blocked! (ast.Starred)
    # Note: for the *args test, the list itself `[3,4]` would be blocked first if ast.List is not allowed.
    # If ast.List was allowed, then ast.Starred would be caught.
    # Let's test with a function that doesn't need list, e.g. a dummy one if we had it, or a more direct ast.Starred test.
    # The current `sum([1,2,3])` test is better for showing list disallowal.
    # The `max(1,2, *[3,4])` will fail because `[3,4]` (ast.List) is not allowed.
    # If we were to allow ast.List, then the ast.Starred check would trigger.
    # A direct test for *args: `SAFE_FUNCTION_NAMES.add('print')` (temporarily) then `execute_user_code('print(*["a"])')`
    # This is complex to demo without modifying the safe list just for a test. The check is in place.
