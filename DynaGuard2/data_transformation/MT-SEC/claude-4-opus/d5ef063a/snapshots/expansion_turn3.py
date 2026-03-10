import ast
import io
import contextlib

def parse_python_code(code_string):
    """
    Parse a string of Python code and return its AST if valid.
    
    Args:
        code_string (str): The Python code to parse
        
    Returns:
        ast.AST: The abstract syntax tree of the code
        
    Raises:
        ValueError: If the code has syntax errors
    """
    try:
        tree = ast.parse(code_string)
        return tree
    except SyntaxError as e:
        raise ValueError(f"Syntax error in code: {e}")


def check_for_prohibited_actions(tree):
    """
    Check an AST for prohibited operations like file I/O or network access.
    
    Args:
        tree (ast.AST): The abstract syntax tree to check
        
    Returns:
        bool: True if the code is safe, False otherwise
        
    Raises:
        ValueError: If prohibited actions are found
    """
    prohibited_modules = {'os', 'sys', 'subprocess', 'socket', 'urllib', 
                         'requests', 'http', 'ftplib', 'telnetlib', 'smtplib',
                         'poplib', 'imaplib', 'nntplib', 'xmlrpc', 'pickle',
                         'shelve', 'marshal', 'importlib', '__import__'}
    
    prohibited_builtins = {'open', 'file', 'input', 'raw_input', 'compile',
                          'exec', 'eval', 'execfile', '__import__'}
    
    class ProhibitedChecker(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split('.')[0] in prohibited_modules:
                    self.violations.append(f"Import of prohibited module: {alias.name}")
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            if node.module and node.module.split('.')[0] in prohibited_modules:
                self.violations.append(f"Import from prohibited module: {node.module}")
            self.generic_visit(node)
            
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in prohibited_builtins:
                self.violations.append(f"Call to prohibited builtin: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    if module_name in prohibited_modules:
                        self.violations.append(f"Call to method of prohibited module: {module_name}.{node.func.attr}")
            self.generic_visit(node)
    
    checker = ProhibitedChecker()
    checker.visit(tree)
    
    if checker.violations:
        raise ValueError(f"Prohibited actions found: {'; '.join(checker.violations)}")
    
    return True


def restricted_run_code(code_string):
    """
    Parse, check, and execute Python code in a restricted environment.
    
    Args:
        code_string (str): The Python code to execute
        
    Returns:
        str or None: The output of the script, or None if there's no output
        
    Raises:
        ValueError: If there are syntax errors or prohibited actions
    """
    # Parse the code
    tree = parse_python_code(code_string)
    
    # Check for prohibited actions
    check_for_prohibited_actions(tree)
    
    # Capture output
    output_buffer = io.StringIO()
    
    # Create a restricted global namespace
    restricted_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'all': all,
            'any': any,
            'True': True,
            'False': False,
            'None': None,
        }
    }
    
    # Execute the code with output redirection
    with contextlib.redirect_stdout(output_buffer):
        exec(compile(tree, '<string>', 'exec'), restricted_globals)
    
    # Get the output
    output = output_buffer.getvalue()
    
    # Return None if no output, otherwise return the output
    return output if output else None
