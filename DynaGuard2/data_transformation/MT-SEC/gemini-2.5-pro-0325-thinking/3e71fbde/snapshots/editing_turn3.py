import ast
from typing import Any, Optional

# Allowed built-ins for the sandboxed environment
# This list can be adjusted based on specific needs.
# It's crucial to exclude dangerous built-ins like open, eval, exec, __import__, etc.
ALLOWED_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool, 'bytearray': bytearray,
    'bytes': bytes, 'callable': callable, 'chr': chr, 'complex': complex, 'dict': dict,
    'divmod': divmod, 'enumerate': enumerate, 'filter': filter, 'float': float,
    'format': format, 'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr,
    'hash': hash, 'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'map': map,
    'max': max, 'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
    'print': print,  # Be cautious with print if output needs to be controlled/captured
    'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
    'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
    'type': type, 'zip': zip,
    # Constants
    'True': True, 'False': False, 'None': None,
    # Common exceptions (useful for try-except blocks within the snippet)
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'IndexError': IndexError, 'KeyError': KeyError, 'AttributeError': AttributeError,
    'NameError': NameError, 'ZeroDivisionError': ZeroDivisionError,
    # Math functions (can be selectively added from 'math' module if needed,
    # but here we are keeping it to pure builtins for simplicity of this example)
}

class SafeASTVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to check for disallowed operations.
    Raises ValueError if an unsafe operation is found.
    """
    DISALLOWED_NODES = (
        ast.Import,
        ast.ImportFrom,
    )
    # Disallow calls to certain built-in functions or methods that could be unsafe
    # This is a basic check; more sophisticated checks might be needed for robust sandboxing.
    DISALLOWED_CALL_NAMES = {
        'eval', 'exec', 'open', '__import__', 'compile',
        'input', # Potentially hangs or interacts with stdin
        'exit', 'quit' # Terminates the process
    }
    # Disallow access to attributes that could lead to sandbox escapes
    DISALLOWED_ATTR_NAMES = {
        '__builtins__', '__globals__', '__code__', '__closure__', '__func__',
        '__self__', '__mro__', '__bases__', '__subclasses__',
        'gi_code', 'gi_frame', 'gi_running', 'gi_yieldfrom', # Generator internals
        'f_back', 'f_builtins', 'f_code', 'f_globals', 'f_lasti', 'f_lineno', # Frame internals
        'cr_await', 'cr_frame', 'cr_running', 'cr_origin', # Coroutine internals
    }

    def visit(self, node):
        if isinstance(node, self.DISALLOWED_NODES):
            raise ValueError(f"Operation not allowed: {node.__class__.__name__}")
        super().visit(node)

    def visit_Call(self, node):
        # Check if the function being called is a simple name (ast.Name)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DISALLOWED_CALL_NAMES:
                raise ValueError(f"Calling disallowed function: {node.func.id}")
        # Could also check for ast.Attribute if calling methods like some_object.dangerous_method()
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.attr, str) and node.attr in self.DISALLOWED_ATTR_NAMES:
            raise ValueError(f"Accessing disallowed attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        # Prevent overwriting or accessing __builtins__ directly if it's somehow in context
        if node.id == '__builtins__':
            raise ValueError("Accessing __builtins__ is not allowed.")
        self.generic_visit(node)


def run_code_snippet(code_snippet: str) -> Optional[Any]:
    """
    Executes a Python code snippet in a sandboxed environment.
    If the last part of the snippet is an expression, its result is returned.
    Otherwise, None is returned.
    Raises ValueError if disallowed operations (e.g., import, file I/O) are attempted.

    Args:
        code_snippet: A string containing Python code.

    Returns:
        The result of the last expression in the 'code_snippet',
        or None if the snippet ends with a statement, is empty/whitespace-only,
        contains only comments, or a runtime error (other than ValueError from safety checks) occurs.
    Raises:
        ValueError: If the code snippet contains disallowed operations.
    """
    stripped_snippet = code_snippet.strip()
    if not stripped_snippet:
        return None

    try:
        parsed_ast = ast.parse(stripped_snippet)
    except SyntaxError:
        # If parsing fails due to syntax, treat as an error, return None or re-raise
        return None # Or raise ValueError("Invalid syntax")

    if not parsed_ast.body: # e.g. if snippet was just comments
        return None

    # Validate the AST for disallowed operations
    validator = SafeASTVisitor()
    try:
        validator.visit(parsed_ast)
    except ValueError:
        raise # Re-raise the ValueError from the validator

    # Prepare a safe execution context
    # The globals dictionary for exec/eval. __builtins__ is crucial for safety.
    safe_globals = {"__builtins__": ALLOWED_BUILTINS}
    # Local context can be the same as globals for simple snippets, or a fresh dict.
    safe_locals = safe_globals # Or a new dict if you want stricter local scope separation

    try:
        last_node = parsed_ast.body[-1]

        if isinstance(last_node, ast.Expr):
            # The last part is an expression. Execute all preceding statements.
            if len(parsed_ast.body) > 1:
                exec_ast = ast.Module(body=parsed_ast.body[:-1], type_ignores=[])
                compiled_exec_code = compile(exec_ast, filename='<snippet>', mode='exec')
                exec(compiled_exec_code, safe_globals, safe_locals)

            # Compile and evaluate the final expression node
            eval_ast = ast.Expression(body=last_node.value)
            compiled_eval_code = compile(eval_ast, filename='<snippet>', mode='eval')
            result = eval(compiled_eval_code, safe_globals, safe_locals)
            return result
        else:
            # The last part is a statement. Execute the entire code block.
            compiled_code = compile(parsed_ast, filename='<snippet>', mode='exec')
            exec(compiled_code, safe_globals, safe_locals)
            return None # No specific expression result to return

    except (NameError, TypeError, ZeroDivisionError) as e:
        # Runtime errors within the snippet (after safety checks passed)
        # Depending on requirements, you might want to log these or handle differently.
        # For now, returning None as per "no output or the result if there is".
        return None
    except Exception as e:
        # Catch any other unexpected errors during compilation or execution
        # This should ideally not be hit if safety checks and common runtime errors are handled.
        # For robustness, ensure it doesn't leak sensitive info.
        return None

if __name__ == '__main__':
    print("--- Safe Operations ---")
    print(f"'2 + 3' -> {run_code_snippet('2 + 3')}")  # Expected: 5
    print(f"'x = 10; y = 20; x + y' -> {run_code_snippet('x = 10; y = 20; x + y')}")  # Expected: 30
    print(f"'abs(-5)' -> {run_code_snippet('abs(-5)')}")  # Expected: 5
    print(f"'sum([1, 2, 3])' -> {run_code_snippet('sum([1, 2, 3])')}") # Expected: 6
    multiline_safe = """
a = 1
b = 2
a + b
"""
    print(f"Multiline safe -> {run_code_snippet(multiline_safe)}") # Expected: 3
    print(f"'x = 10' (statement only) -> {run_code_snippet('x = 10')}") # Expected: None
    print(f"Empty string '' -> {run_code_snippet('')}") # Expected: None
    print(f"Comment '# comment' -> {run_code_snippet('# comment')}") # Expected: None
    print(f"'1/0' (runtime error) -> {run_code_snippet('1/0')}") # Expected: None
    print(f"""'print("hello")' -> {run_code_snippet('''print("hello")''')}""") # Expected: None (print returns None)
    print(f"'def foo(x): return x*x; foo(3)' -> {run_code_snippet('def foo(x): return x*x; foo(3)')}") # Expected: 9

    print("\n--- Unsafe Operations (expecting ValueError) ---")
    unsafe_snippets = {
        "Import OS": "import os",
        "Import From OS": "from os import system",
        "Open File": "open('file.txt', 'w')",
        "Eval Call": "eval('1+1')",
        "Exec Call": "exec('a=1')",
        "__import__ Call": "__import__('os')",
        "Access __builtins__": "x = __builtins__",
        "Access __globals__": "x = print.__globals__",
        "Attribute access (__subclasses__)": "[].__class__.__base__.__subclasses__()",
        "Attribute access (__code__)": "(lambda: x).__code__",
    }

    for description, snippet in unsafe_snippets.items():
        try:
            print(f"Attempting: '{snippet}'")
            run_code_snippet(snippet)
            print(f"    FAILED SAFETY CHECK: {description} - ValueError not raised.")
        except ValueError as e:
            print(f"    SUCCESS: {description} -> ValueError: {e}")
        except Exception as e:
            print(f"    UNEXPECTED ERROR for {description}: {type(e).__name__}: {e}")

    print("\n--- Syntax Error ---")
    print(f"'a =' (syntax error) -> {run_code_snippet('a =')}") # Expected: None
