import ast

def analyze_ast_for_safety(code_string: str) -> bool:
    """
    Parses a string of Python code into an AST and checks for
    potentially harmful operations.

    Args:
        code_string: The Python code to analyze.

    Returns:
        True if the script is deemed safe, False otherwise.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # If the code can't be parsed, it can't be run,
        # but we'll consider it "unsafe" from an execution standpoint.
        return False

    for node in ast.walk(tree):
        # Disallow imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        
        # Disallow calls to certain built-in functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['open', 'eval', 'exec']:
                    return False
            # Disallow attribute calls like os.system
            if isinstance(node.func, ast.Attribute):
                # This is a simple check, could be more sophisticated
                # e.g., checking if 'os' or 'subprocess' was imported and then used.
                # For now, let's be very restrictive.
                # A more robust check would track imported modules.
                if node.func.attr in ['system', 'popen', 'spawn', 'call', 'run']:
                    return False

    return True

if __name__ == '__main__':
    safe_code_1 = "print('Hello, world!')"
    safe_code_2 = "x = 1 + 2\nprint(x)"
    safe_code_3 = """
def greet(name):
    return f"Hello, {name}"
print(greet("User"))
"""

    unsafe_code_1 = "import os\nos.system('echo unsafe')"
    unsafe_code_2 = "open('some_file.txt', 'w').write('dangerous')"
    unsafe_code_3 = "eval('1 + 1')" # eval itself is disallowed
    unsafe_code_4 = "exec('print(\"executed\")')" # exec itself is disallowed
    unsafe_code_5 = "__import__('os').system('echo unsafe')" # using __import__
    unsafe_code_6 = "import subprocess\nsubprocess.call(['ls'])"
    unsafe_code_7 = """
import os
def do_something():
    os.system("echo 'dangerous'")
do_something()
"""

    print(f"'{safe_code_1[:20]}...' is safe: {analyze_ast_for_safety(safe_code_1)}")
    print(f"'{safe_code_2[:20]}...' is safe: {analyze_ast_for_safety(safe_code_2)}")
    print(f"'{safe_code_3[:20]}...' is safe: {analyze_ast_for_safety(safe_code_3)}")

    print(f"'{unsafe_code_1[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_1)}")
    print(f"'{unsafe_code_2[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_2)}")
    print(f"'{unsafe_code_3[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_3)}")
    print(f"'{unsafe_code_4[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_4)}")
    print(f"'{unsafe_code_5[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_5)}")
    print(f"'{unsafe_code_6[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_6)}")
    print(f"'{unsafe_code_7[:20]}...' is safe: {analyze_ast_for_safety(unsafe_code_7)}")

    # Test syntax error
    syntax_error_code = "print('Hello'"
    print(f"'{syntax_error_code[:20]}...' is safe: {analyze_ast_for_safety(syntax_error_code)}")
