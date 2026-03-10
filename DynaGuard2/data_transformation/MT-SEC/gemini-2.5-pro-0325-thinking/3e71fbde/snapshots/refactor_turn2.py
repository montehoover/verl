import ast

class SecurityValidator(ast.NodeVisitor):
    """
    AST visitor to check for disallowed operations in a Python code snippet.
    """
    def __init__(self):
        super().__init__()
        self.disallowed_found = False
        self.error_message = ""

        # Disallowed node types
        self.disallowed_node_types = (
            ast.Import,
            ast.ImportFrom,
            # Potentially others like ast.Delete for critical names, ast.Exec
        )
        # Disallowed names for functions called directly
        self.disallowed_call_names = {'open', 'eval', 'exec', '__import__'}
        # Disallowed attribute names (e.g., obj.attr)
        self.disallowed_attribute_names = {
            '__builtins__', '__class__', '__subclasses__', '__globals__',
            '__code__', '__mro__', '__bases__', 'gi_frame', 'gi_code',
            'f_locals', 'f_globals', 'f_builtins', 'f_code',
            '__dict__', # Can be used to bypass restrictions
            '__getattribute__', '__setattr__', '__delattr__', # Overriding these can be risky
            '__init__', # Accessing __init__ of sensitive objects
            '__call__', # Potentially risky depending on context
        }

    def visit(self, node):
        if self.disallowed_found: # Stop visiting if already found an issue
            return
        
        node_type = type(node)
        if node_type in self.disallowed_node_types:
            self.disallowed_found = True
            self.error_message = f"Disallowed operation: {node_type.__name__}"
            return
        super().visit(node)

    def visit_Call(self, node):
        if self.disallowed_found:
            return
        # Check for disallowed function calls by name
        if isinstance(node.func, ast.Name) and node.func.id in self.disallowed_call_names:
            self.disallowed_found = True
            self.error_message = f"Disallowed function call: {node.func.id}"
            return
        # Check for calls on disallowed attributes, e.g. ().__class__.mro()
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.disallowed_attribute_names:
            self.disallowed_found = True
            self.error_message = f"Call on disallowed attribute: {node.func.attr}"
            return
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.disallowed_found:
            return
        # Check for access to disallowed attribute names
        if node.attr in self.disallowed_attribute_names:
            self.disallowed_found = True
            self.error_message = f"Disallowed attribute access: {node.attr}"
            return
        self.generic_visit(node)

# Define a restricted set of allowed built-in functions
ALLOWED_BUILTINS = {
    'print': print,
    'len': len,
    'range': range,
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'None': None,
    'True': True,
    'False': False,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'type': type,
    'round': round,
    'sorted': sorted,
    'zip': zip,
    'enumerate': enumerate,
    'map': map,
    'filter': filter,
    'all': all,
    'any': any,
    'chr': chr,
    'ord': ord,
    'pow': pow,
    'repr': repr,
    # Errors are generally safe to be available
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'NameError': NameError,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'AttributeError': AttributeError,
    'SyntaxError': SyntaxError, # Though snippet parsing catches this first
    'ZeroDivisionError': ZeroDivisionError,
}

def _parse_and_validate_snippet(snippet: str) -> ast.AST:
    """
    Parses a Python code snippet string into an AST and validates it.

    Args:
        snippet: A string containing the Python code to parse.

    Returns:
        The parsed and validated AST object.

    Raises:
        ValueError: If the code snippet has a syntax error or contains
                    disallowed operations.
    """
    try:
        tree = ast.parse(snippet)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in snippet: {e}")

    validator = SecurityValidator()
    validator.visit(tree)
    if validator.disallowed_found:
        raise ValueError(f"Disallowed operation found in code snippet: {validator.error_message}")
    return tree

def _execute_ast(tree: ast.AST, restricted_globals: dict, local_vars: dict):
    """
    Executes a validated AST in a restricted environment.

    Args:
        tree: The AST object to execute.
        restricted_globals: The globals dictionary for execution.
        local_vars: The locals dictionary for execution.

    Returns:
        The result of the last expression in the AST, or None if
        the AST has no final expression or is empty.
    
    Raises:
        RuntimeError: If an error occurs during code execution.
    """
    if not tree.body:  # Empty snippet or only comments
        return None

    # If the last statement is an expression, compile and eval it separately
    # to get its value. Otherwise, exec all statements.
    if isinstance(tree.body[-1], ast.Expr):
        # Compile and execute all statements before the last one
        if len(tree.body) > 1:
            exec_statements_ast = ast.Module(body=tree.body[:-1], type_ignores=[])
            try:
                compiled_exec_code = compile(exec_statements_ast, filename="<snippet_exec>", mode="exec")
                exec(compiled_exec_code, restricted_globals, local_vars)
            except Exception as e:
                # Catch runtime errors from the exec part
                raise RuntimeError(f"Error executing snippet: {e}")
        
        # Compile and evaluate the last expression statement
        last_expr_ast = ast.Expression(body=tree.body[-1].value)
        try:
            compiled_eval_code = compile(last_expr_ast, filename="<snippet_eval>", mode="eval")
            result = eval(compiled_eval_code, restricted_globals, local_vars)
            return result
        except Exception as e:
            # Catch runtime errors from the eval part
            raise RuntimeError(f"Error evaluating final expression in snippet: {e}")
    else:
        # All statements are non-expressions, or the last one isn't an expression.
        # Compile and execute the whole snippet.
        try:
            compiled_code = compile(tree, filename="<snippet>", mode="exec")
            exec(compiled_code, restricted_globals, local_vars)
            return None # No specific result to return from exec
        except Exception as e:
            # Catch runtime errors from the exec part
            raise RuntimeError(f"Error executing snippet: {e}")

def run_code_snippet(snippet: str):
    """
    Executes a Python code snippet in a restricted environment.

    Args:
        snippet: A string containing the Python code to execute.

    Returns:
        The result of the last expression in the snippet, or None if
        the snippet has no final expression or is empty.

    Raises:
        ValueError: If the code snippet contains disallowed operations
                    or has a syntax error.
        RuntimeError: If an error occurs during code execution.
    """
    # Step 1: Parse and validate the snippet
    # This can raise ValueError
    tree = _parse_and_validate_snippet(snippet)

    # Step 2: Prepare the execution environment
    restricted_globals = {"__builtins__": ALLOWED_BUILTINS.copy()}
    # One could add safe modules here, e.g., a curated 'math' module.
    # restricted_globals['math'] = math # If math was imported and deemed safe.
    local_vars = {}

    # Step 3: Execute the AST
    # This can raise RuntimeError
    return _execute_ast(tree, restricted_globals, local_vars)

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("Running examples for run_code_snippet:")

    snippets_to_test = [
        ("Simple arithmetic", "1 + 2"),
        ("Variable assignment", "a = 10\nb = 20\na * b"),
        ("Multi-line, no return", "x = 5\ny = x + 3\nprint(y)"), # print returns None
        ("Function definition and call", "def add(x, y):\n  return x + y\nadd(3, 4)"),
        ("List comprehension", "[i*i for i in range(5)]"),
        ("Empty snippet", ""),
        ("Comment only", "# This is a comment"),
        ("Snippet with print", "print('Hello, world!')"), # print returns None
        ("Allowed built-in usage", "len([1, 2, 3])"),
        ("Disallowed import", "import os"),
        ("Disallowed function call", "open('file.txt')"),
        ("Disallowed attribute access", "().__class__"),
        ("Code with runtime error", "1 / 0"),
        ("Code with NameError", "print(undefined_var)"),
        ("Accessing __builtins__ directly (should be safe dict)", "__builtins__"),
        ("Using a disallowed attribute in a call", "int.__dict__"),
    ]

    for description, snippet_code in snippets_to_test:
        print(f"\n--- Testing: {description} ---")
        print(f"Code:\n{snippet_code}")
        try:
            output = run_code_snippet(snippet_code)
            print(f"Output: {output}")
            if description == "Accessing __builtins__ directly (should be safe dict)":
                if isinstance(output, dict) and 'print' in output and 'open' not in output:
                    print("Output for __builtins__ is the safe dictionary as expected.")
                else:
                    print("Output for __builtins__ is NOT the safe dictionary.")
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")
