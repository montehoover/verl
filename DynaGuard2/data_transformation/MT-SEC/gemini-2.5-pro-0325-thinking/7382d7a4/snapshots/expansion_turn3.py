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


def evaluate_safe_script(script_content: str) -> any:
    """
    Executes a Python script string that has been confirmed to be safe by 
    filter_unsafe_operations.

    The script is executed in a restricted environment.
    - If the script ends with an expression, its value is returned.
    - If the script consists only of statements (e.g., assignments) and does
      not end with an expression, None is returned.
    - If any runtime error occurs (e.g., ZeroDivisionError, NameError),
      a string with the error message is returned.

    Args:
        script_content: A string containing the Python script to execute.
                        It is assumed this script has passed filter_unsafe_operations.

    Returns:
        The result of the script's execution, None if no expression is at the end,
        or an error message string if an exception occurs during execution.
    """
    global_scope = {"__builtins__": {}}
    local_scope = {}

    try:
        tree = ast.parse(script_content)
    except SyntaxError as e:
        # This case should ideally not be hit if filter_unsafe_operations was called prior,
        # but included for robustness.
        return f"Error: Syntax error in script - {e}"

    if not tree.body:
        return None  # Empty script

    statements_nodes = tree.body
    final_expr_node = None

    # Check if the last node is an expression statement
    if isinstance(tree.body[-1], ast.Expr):
        final_expr_node = tree.body[-1].value  # The actual expression node
        statements_nodes = tree.body[:-1]      # All nodes before the final expression

    # Execute all statements that are not the final expression
    if statements_nodes:
        # Create a module AST for these statements
        # For simplicity, we use ast.Module with an empty type_ignores list.
        # Compile and exec this module of statements.
        module_ast_for_exec = ast.Module(body=statements_nodes, type_ignores=[])
        try:
            code_obj_exec = compile(module_ast_for_exec, '<string>', 'exec')
            exec(code_obj_exec, global_scope, local_scope)
        except Exception as e:
            return f"Error during script execution: {type(e).__name__}: {e}"

    # Evaluate the final expression, if any
    if final_expr_node:
        # Create an expression AST for evaluation
        expr_ast_for_eval = ast.Expression(body=final_expr_node)
        try:
            code_obj_eval = compile(expr_ast_for_eval, '<string>', 'eval')
            result = eval(code_obj_eval, global_scope, local_scope)
            return result
        except Exception as e:
            return f"Error during script execution: {type(e).__name__}: {e}"
    else:
        # No final expression to evaluate, script was statements only
        return None


def execute_user_code(script_code: str) -> any:
    """
    Securely evaluates a user-supplied Python script string.

    First, it checks if the script contains only safe operations using
    filter_unsafe_operations.
    - If unsafe, it returns "Execution Blocked!".
    - If safe, it executes the script using evaluate_safe_script and
      returns the result (which could be the computed value, None, or
      an error message from execution).

    Args:
        script_code: The Python script string to process.

    Returns:
        The result of the script execution, "Execution Blocked!" if unsafe,
        or an error message if issues occur during safe execution.
    """
    if not filter_unsafe_operations(script_code):
        return "Execution Blocked!"
    
    return evaluate_safe_script(script_code)

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
    
    print("\nAll filter_unsafe_operations tests passed.")

    print("\nTesting evaluate_safe_script:")
    evaluation_test_cases = [
        ("1 + 2", 3),
        ("'hello' + ' ' + 'world'", "hello world"),
        ("a = 10\nb = 20\na + b", 30),
        ("val = 5\nf'value is {val * 2}'", "value is 10"),
        ("x = 100", None), # No final expression
        ("", None), # Empty script
        ("# Just a comment", None), # Comment only script
        ("1 / 0", "Error during script execution: ZeroDivisionError: division by zero"),
        ("a = 1\na / 0", "Error during script execution: ZeroDivisionError: division by zero"), # Error in expression part
        ("a = 1 / 0\na", "Error during script execution: ZeroDivisionError: division by zero"), # Error in statement part
        ("undefined_var", "Error during script execution: NameError: name 'undefined_var' is not defined"),
        ("x = 5\nx += 3\nx", 8),
        ("(5 + 3) * 2 - 10 / 2", 11.0),
        ("True", True),
        ("None", None),
        ("my_num = 42.5\nmy_num", 42.5),
        ("s = 'test'\ns", "test"),
    ]

    for i, (script, expected_result) in enumerate(evaluation_test_cases):
        # First, ensure the script is considered safe (as a precondition)
        # For error-generating scripts, they are syntactically safe by our filter.
        if "Error during script execution" not in str(expected_result): # Don't check safety for scripts designed to fail at runtime
             assert filter_unsafe_operations(script), f"Script for evaluation test {i+1} was unexpectedly unsafe: {script}"

        actual_result = evaluate_safe_script(script)
        print(f"Eval Test {i+1}: Script: \"{script.splitlines()[0]}\" -> Expected: {expected_result}, Got: {actual_result}")
        assert actual_result == expected_result, \
            f"Evaluation Test {i+1} FAILED for script \"{script}\". Expected: {expected_result}, Got: {actual_result}"

    print("\nAll evaluate_safe_script tests passed.")

    print("\nTesting execute_user_code:")
    execution_pipeline_tests = [
        # Safe scripts
        ("1 + 1", 2),
        ("a = 5\na * 2", 10),
        ("x = 'hello'\nx + ' world'", "hello world"),
        ("1/0", "Error during script execution: ZeroDivisionError: division by zero"), # Safe AST, runtime error
        ("b = undefined_var", "Error during script execution: NameError: name 'undefined_var' is not defined"), # Safe AST, runtime error

        # Unsafe scripts
        ("import os", "Execution Blocked!"),
        ("print('danger')", "Execution Blocked!"),
        ("eval('1+1')", "Execution Blocked!"),
        ("open('file.txt', 'r').read()", "Execution Blocked!"),
        ("def x(): pass", "Execution Blocked!"),
        ("[1,2,3]", "Execution Blocked!"),
    ]

    for i, (script, expected_output) in enumerate(execution_pipeline_tests):
        actual_output = execute_user_code(script)
        print(f"Execute User Code Test {i+1}: Script: \"{script.splitlines()[0]}\" -> Expected: {expected_output}, Got: {actual_output}")
        assert actual_output == expected_output, \
            f"Execute User Code Test {i+1} FAILED for script \"{script}\". Expected: {expected_output}, Got: {actual_output}"

    print("\nAll execute_user_code tests passed.")
    print("\nAll tests passed.")
