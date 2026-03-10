import re
import ast

# Define a set of security-sensitive characters.
# These are characters often associated with command injection or other vulnerabilities.
SECURITY_SENSITIVE_CHARS = set([';', '&', '|', '<', '>', '`', '$', '(', ')', '{', '}', '[', ']', '"', "'", '\\'])
# Allow specific punctuation if needed, e.g. for file paths or specific DSLs
# For now, we'll be quite restrictive.
# Alphanumeric tokens are generally safe. Other symbols might be context-dependent.

# Define a set of approved syntax elements (tokens).
# For real-world scenarios, this list would be more comprehensive or rule-based.
APPROVED_SYNTAX_ELEMENTS = {
    "hello", "world",
    "run_analysis", "--input", "data.csv", "--output", "results.txt",
    "copy", "file1.txt", "/tmp/file2.new",
    "justalphanumericonly",
    "with_underscores_and_numbers123",
    "path-with-hyphens",
    # Add any other specific tokens that should be considered valid
}

def analyze_script(script: str) -> list[str]:
    """
    Parses a script, tokenizes it, and validates each token against a list
    of approved syntax elements. Returns the list of tokens if all are valid.

    Raises ValueError if the script contains security-sensitive characters
    or if any token is not in the approved list.

    Args:
        script: The script string to analyze.

    Returns:
        A list of tokens from the script, if all tokens are validated.

    Raises:
        ValueError: If the script contains security-sensitive characters,
                    or if any token is not an approved syntax element.
        TypeError: If the input script is not a string.
    """
    if not isinstance(script, str):
        raise TypeError("Input script must be a string.")

    # 1. Check for security-sensitive characters in the entire script
    for char in script:
        if char in SECURITY_SENSITIVE_CHARS:
            raise ValueError(f"Script contains security-sensitive character: '{char}'")

    # 2. Tokenize the script.
    tokens = script.split()

    if not tokens: # Handles empty script or script with only whitespace
        return []

    # 3. Validate each token against the approved list
    for token in tokens:
        if token not in APPROVED_SYNTAX_ELEMENTS:
            raise ValueError(f"Token '{token}' is not an approved syntax element.")
    
    return tokens # If all tokens passed validation


# --- run_user_script function and helpers ---

DISALLOWED_NODES = (
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Delete,
    ast.With, ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.Yield, ast.YieldFrom,
    ast.Try,
    ast.Global, ast.Nonlocal,
)

DISALLOWED_BUILTIN_CALLS = {
    'eval', 'exec', 'execfile', 'open', 'compile', '__import__',
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'input', 'breakpoint', 'memoryview',
}

SAFE_DUNDERS = frozenset([
    '__bool__', '__int__', '__float__', '__str__', '__repr__', '__hash__',
    '__len__', '__getitem__', '__setitem__', '__delitem__', '__iter__', '__next__', '__contains__',
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__',
    '__radd__', '__rsub__', '__rmul__', '__rtruediv__', '__rfloordiv__', '__rmod__', '__rpow__',
    '__iadd__', '__isub__', '__imul__', '__itruediv__', '__ifloordiv__', '__imod__', '__ipow__',
    '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__call__', '__abs__', '__round__',
])

SAFE_BUILTINS = {
    'print': print, 'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
    'True': True, 'False': False, 'None': None,
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow, 'divmod': divmod,
    'range': range, 'zip': zip, 'enumerate': enumerate, 'map': map, 'filter': filter,
    'sorted': sorted, 'all': all, 'any': any,
    'isinstance': isinstance,
    'Exception': Exception, # Allow raising generic exceptions from the script
}

def run_user_script(user_script: str):
    """
    Safely executes a user-provided script string.

    The script is parsed and checked for disallowed operations (e.g., imports,
    file access, defining functions/classes, calls to dangerous builtins,
    access to certain dunder attributes).

    If the script ends with an expression, its result is returned after executing
    any preceding statements. If the script does not end with an expression
    (e.g., ends with an assignment or is empty after parsing), None is returned.

    Args:
        user_script: The Python script string to execute.

    Returns:
        The result of the script's final expression, or None.

    Raises:
        ValueError: If the script contains syntax errors, disallowed operations,
                    or if an error occurs during execution within the sandboxed environment.
        TypeError: If user_script is not a string.
    """
    if not isinstance(user_script, str):
        raise TypeError("Input script must be a string.")

    try:
        tree = ast.parse(user_script, filename='<user_script>')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            raise ValueError(f"Disallowed language feature: {type(node).__name__}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_BUILTIN_CALLS:
                raise ValueError(f"Disallowed function call: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') and node.attr.endswith('__') and len(node.attr) > 4:
                if node.attr not in SAFE_DUNDERS:
                    raise ValueError(f"Access to disallowed attribute: {node.attr}")

    exec_globals = {"__builtins__": SAFE_BUILTINS}
    exec_locals = {} 

    try:
        if not tree.body: 
            return None # Empty script or only comments

        # If the last node is an expression, it might be the return value
        if isinstance(tree.body[-1], ast.Expr):
            last_expr_value_node = tree.body.pop().value # The actual expression node (e.g., ast.BinOp)
            
            # Execute all preceding statements if any
            if tree.body:
                module_statements_only = ast.Module(body=tree.body, type_ignores=[])
                compiled_statements = compile(module_statements_only, '<user_script_stmts>', 'exec')
                exec(compiled_statements, exec_globals, exec_locals)
            
            # Evaluate the final expression in the context of the executed statements
            final_expr_to_eval = ast.Expression(body=last_expr_value_node)
            compiled_expr = compile(final_expr_to_eval, '<user_script_expr>', 'eval')
            return eval(compiled_expr, exec_globals, exec_locals)
        else:
            # The script does not end with an expression (e.g., ends with assignment, def, etc.)
            # Or it was empty and handled already.
            compiled_code = compile(tree, filename='<user_script_all_stmts>', mode='exec')
            exec(compiled_code, exec_globals, exec_locals)
            return None # exec itself returns None
    except Exception as e:
        raise ValueError(f"Error during script execution: {type(e).__name__}: {e}") from e

# --- End of run_user_script ---

if __name__ == '__main__':
    # Example Usage
    # Note: APPROVED_SYNTAX_ELEMENTS is defined globally above.

    print("--- Testing valid scripts ---")
    # Valid scripts (all tokens must be in APPROVED_SYNTAX_ELEMENTS and no security chars)
    valid_scripts = [
        "hello world",
        "run_analysis --input data.csv --output results.txt",
        "copy file1.txt /tmp/file2.new",
        "  leading and trailing spaces  ", # results in approved tokens or empty list
        "", # Empty script
        "justalphanumericonly",
        "with_underscores_and_numbers123",
        "path-with-hyphens",
    ]

    for s in valid_scripts:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}")
        except (ValueError, TypeError) as e:
            print(f"Script: '{s}' -> ERROR: {e}") # Should not happen for these

    print("\n--- Testing invalid scripts (unapproved tokens) ---")
    invalid_scripts_unapproved_token = [
        "hello unknown_token", # 'unknown_token' is not in APPROVED_SYNTAX_ELEMENTS
        "run_analysis --input data.csv --output results.txt --verbose", # '--verbose' is not in APPROVED_SYNTAX_ELEMENTS
        "copy file1.txt /tmp/file2.new --force", # '--force' is not in APPROVED_SYNTAX_ELEMENTS
        "a_brand_new_command" # 'a_brand_new_command' is not in APPROVED_SYNTAX_ELEMENTS
    ]

    for s in invalid_scripts_unapproved_token:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}") # Should not reach here
        except (ValueError, TypeError) as e: # Expect ValueError here
            print(f"Script: '{s}' -> ERROR: {e}")

    print("\n--- Testing invalid scripts (security-sensitive characters) ---")
    # These should fail before token validation, due to character check.
    invalid_scripts_security = [
        "echo 'hello world'",        # Contains '
        "cat /etc/passwd | grep root", # Contains |
        "rm -rf / &",                # Contains &
        "command `uname -a`",        # Contains `
        "script_with_semicolon;",    # Contains ;
        "test_script --option \"quoted string\"", # Contains "
        "some_command < input.txt > output.txt", # Contains < or >
        "variable=$HOME",            # Contains $
        "array_access[0]",           # Contains [ or ]
        "code_block{echo hi}",       # Contains { or }
        "path_with_backslash\\test", # Contains \
        "call_function(arg1)",       # Contains ( or )
    ]

    for s in invalid_scripts_security:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}")
        except (ValueError, TypeError) as e:
            print(f"Script: '{s}' -> ERROR: {e}")

    print("\n--- Testing type error ---")
    invalid_inputs_type = [
        123,
        None,
        ["list", "is", "not", "string"],
    ]
    for s_invalid in invalid_inputs_type:
        try:
            analyze_script(s_invalid)
            print(f"Script: {s_invalid} -> Tokens: {tokens}") # Should not reach here
        except TypeError as e:
            print(f"Script: {s_invalid} ({type(s_invalid).__name__}) -> ERROR: {e}")
        except ValueError as e: # Should ideally be caught by TypeError first
            print(f"Script: {s_invalid} ({type(s_invalid).__name__}) -> UNEXPECTED ValueError: {e}")

    print("\n\n--- Testing run_user_script ---")

    test_cases_run_script = [
        # Valid scripts and expected results
        ("2 + 2", 4),
        ("x = 10\ny = 20\nx * y", 200),
        ("len([1, 2, 3])", 3),
        ("a = [1,2,3]\na.append(4)\nlen(a)", 4),
        ("print('hello from script')", None), # print itself returns None, script ends with statement
        ("print('hello from script')\n42", 42), # print executes, last expr is 42
        ("10", 10),
        ("sum([i for i in range(3)])", 3), # 0+1+2
        ("list(map(lambda x: x*x, [1,2,3]))", [1,4,9]),
        ("x = 10\nif x > 5:\n  'greater'\nelse:\n  'less'", 'greater'),
        ("y = None\nif False:\n  y = 1\ny", None),
        ("'hello'.upper()", "HELLO"),
        ("z = {'a': 1}\nz['a']", 1),
        ("# Just a comment\n42", 42),
        ("\n# Only comments and whitespace\n  \n", None), # Empty script effectively
        ("x = 1\nx = 2\nx", 2), # Last expression is x
        ("x=1", None), # Ends with assignment

        # Scripts that should raise ValueError (SyntaxError)
        ("2 +", ValueError("Syntax error in script: unexpected EOF while parsing")),
        ("x = ", ValueError("Syntax error in script: invalid syntax")),

        # Scripts that should raise ValueError (Disallowed Node)
        ("import os", ValueError("Disallowed language feature: Import")),
        ("from math import sqrt", ValueError("Disallowed language feature: ImportFrom")),
        ("def foo(): pass", ValueError("Disallowed language feature: FunctionDef")),
        ("class MyClass: pass", ValueError("Disallowed language feature: ClassDef")),
        ("del x", ValueError("Disallowed language feature: Delete")), # Assuming x could exist
        ("try:\n  1/0\nexcept:\n  pass", ValueError("Disallowed language feature: Try")),
        ("with open('f.txt') as f: pass", ValueError("Disallowed language feature: With")), # open is also disallowed call

        # Scripts that should raise ValueError (Disallowed Call)
        ("eval('1+1')", ValueError("Disallowed function call: eval")),
        ("open('file.txt')", ValueError("Disallowed function call: open")),
        ("__import__('os')", ValueError("Disallowed function call: __import__")),

        # Scripts that should raise ValueError (Disallowed Attribute Access)
        ("''.__class__", ValueError("Access to disallowed attribute: __class__")),
        ("(lambda: 1).__globals__", ValueError("Access to disallowed attribute: __globals__")),
        ("x = (1).__dict__", ValueError("Access to disallowed attribute: __dict__")), # Example, int has no __dict__ but check is generic

        # Scripts that should raise ValueError (Execution Error)
        ("1 / 0", ValueError("Error during script execution: ZeroDivisionError: division by zero")),
        ("non_existent_var", ValueError("Error during script execution: NameError: name 'non_existent_var' is not defined")),
        ("x = [1]\nx[5]", ValueError("Error during script execution: IndexError: list index out of range")),
        ("print(unknown_var)", ValueError("Error during script execution: NameError: name 'unknown_var' is not defined")),

        # Type Error for run_user_script input
        (123, TypeError("Input script must be a string.")),
        (None, TypeError("Input script must be a string.")),
    ]

    for i, (script, expected) in enumerate(test_cases_run_script):
        print(f"\nTest {i+1}: Script = {repr(script)}")
        try:
            result = run_user_script(script)
            print(f"  Result: {repr(result)}")
            if isinstance(expected, Exception):
                print(f"  FAILED: Expected Exception {type(expected).__name__}, but got result.")
            elif result != expected:
                print(f"  FAILED: Expected {repr(expected)}, but got {repr(result)}.")
            else:
                print("  PASSED.")
        except (ValueError, TypeError) as e:
            print(f"  Got Exception: {type(e).__name__}: {e}")
            if isinstance(expected, Exception) and type(e) == type(expected) and str(e) == str(expected):
                print("  PASSED (expected exception).")
            elif isinstance(expected, Exception):
                print(f"  FAILED: Expected Exception {type(expected).__name__} ('{str(expected)}'), but got {type(e).__name__} ('{str(e)}').")
            else:
                print(f"  FAILED: Expected result {repr(expected)}, but got Exception {type(e).__name__}.")
        except Exception as e: # Catch any other unexpected exceptions
            print(f"  UNEXPECTED Exception: {type(e).__name__}: {e}")
            if isinstance(expected, Exception):
                print(f"  FAILED: Expected Exception {type(expected).__name__}, but got {type(e).__name__}.")
            else:
                print(f"  FAILED: Expected result {repr(expected)}, but got unexpected Exception.")
