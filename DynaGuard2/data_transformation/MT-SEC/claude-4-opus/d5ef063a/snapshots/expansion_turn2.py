import ast

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
