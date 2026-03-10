import ast
import types
import sys
import io

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

class SafetyAnalyzer(ast.NodeVisitor):
    """Visitor to analyze script safety"""
    
    def __init__(self):
        self.is_safe = True
        self.unsafe_operations = []
    
    def visit_Import(self, node):
        self.is_safe = False
        self.unsafe_operations.append('import')
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.is_safe = False
        self.unsafe_operations.append('import from')
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Check for unsafe built-in functions
        unsafe_functions = {
            'exec', 'eval', '__import__', 'compile', 'open',
            'file', 'input', 'raw_input', 'execfile',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'globals', 'locals', 'vars', 'dir'
        }
        
        if isinstance(node.func, ast.Name) and node.func.id in unsafe_functions:
            self.is_safe = False
            self.unsafe_operations.append(f'{node.func.id}()')
        
        # Check for attribute access that might be file operations
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ['read', 'write', 'open', 'close', 'seek', 'tell']:
                self.is_safe = False
                self.unsafe_operations.append(f'.{node.func.attr}()')
        
        self.generic_visit(node)
    
    def visit_With(self, node):
        # Check for context managers that might be file operations
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Name):
                    if item.context_expr.func.id in ['open', 'file']:
                        self.is_safe = False
                        self.unsafe_operations.append('with open/file')
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        # Check for dangerous decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['eval', 'exec']:
                    self.is_safe = False
                    self.unsafe_operations.append('@eval/@exec decorator')
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        # Check for dangerous decorators on classes
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['eval', 'exec']:
                    self.is_safe = False
                    self.unsafe_operations.append('@eval/@exec decorator')
        self.generic_visit(node)

def analyze_script_safety(tree):
    """Analyze an AST to determine if the script is safe to execute"""
    analyzer = SafetyAnalyzer()
    analyzer.visit(tree)
    return analyzer.is_safe

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

def execute_user_script(script):
    """Execute a user's script safely after verification"""
    try:
        # Parse the script into an AST
        tree = ast.parse(script, mode='exec')
    except SyntaxError as e:
        raise ValueError(f"Syntax error in script: {e}")
    
    # Analyze script safety
    is_safe = analyze_script_safety(tree)
    if not is_safe:
        raise ValueError("Script contains unsafe operations and cannot be executed")
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # Execute in restricted environment
        execute_fn = setup_execution_environment()
        execute_fn(script)
        
        # Get captured output
        output = sys.stdout.getvalue()
        return output
    except SecurityError as e:
        raise ValueError(f"Security violation: {e}")
    except Exception as e:
        raise ValueError(f"Execution error: {e}")
    finally:
        # Restore stdout
        sys.stdout = old_stdout

class SecurityError(Exception):
    """Exception raised for security violations in restricted code"""
    pass
