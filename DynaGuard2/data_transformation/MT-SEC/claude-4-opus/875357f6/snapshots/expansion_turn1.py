import ast
import types

class RestrictedNodeVisitor(ast.NodeVisitor):
    """Visitor to check for restricted operations in the AST"""
    
    def __init__(self):
        self.violations = []
    
    def visit_Import(self, node):
        self.violations.append(f"Import statement not allowed: line {node.lineno}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.violations.append(f"Import from statement not allowed: line {node.lineno}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['exec', 'eval', '__import__', 'compile', 'open']:
                self.violations.append(f"Forbidden function '{node.func.id}' called: line {node.lineno}")
        self.generic_visit(node)

def setup_execution_environment():
    """Initialize a restricted Python execution environment"""
    
    # Create a restricted globals dictionary with safe built-ins only
    safe_builtins = {
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
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'sorted': sorted,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'True': True,
        'False': False,
        'None': None,
    }
    
    restricted_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        '__doc__': None,
    }
    
    def execute_restricted_code(code_string):
        """Execute Python code in the restricted environment"""
        try:
            # Parse the code into an AST
            tree = ast.parse(code_string, mode='exec')
            
            # Check for violations using the visitor
            visitor = RestrictedNodeVisitor()
            visitor.visit(tree)
            
            if visitor.violations:
                raise SecurityError("Code contains restricted operations:\n" + "\n".join(visitor.violations))
            
            # Compile and execute the code in the restricted environment
            compiled_code = compile(tree, '<restricted>', 'exec')
            exec(compiled_code, restricted_globals)
            
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax: {e}")
        except Exception as e:
            raise e
    
    return execute_restricted_code

class SecurityError(Exception):
    """Exception raised for security violations in restricted code"""
    pass
