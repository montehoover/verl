import ast
import re

def parse_script_operations(script_string):
    """
    Parse a Python script string and return a list of operations it contains.
    Operations are limited to basic arithmetic and string manipulations.
    
    Args:
        script_string (str): The Python script as a string
        
    Returns:
        list: A list of operation types found in the script
    """
    operations = []
    
    try:
        tree = ast.parse(script_string)
    except SyntaxError:
        return []
    
    class OperationVisitor(ast.NodeVisitor):
        def visit_BinOp(self, node):
            # Arithmetic operations
            if isinstance(node.op, ast.Add):
                operations.append("addition")
            elif isinstance(node.op, ast.Sub):
                operations.append("subtraction")
            elif isinstance(node.op, ast.Mult):
                operations.append("multiplication")
            elif isinstance(node.op, ast.Div):
                operations.append("division")
            elif isinstance(node.op, ast.FloorDiv):
                operations.append("floor_division")
            elif isinstance(node.op, ast.Mod):
                operations.append("modulo")
            elif isinstance(node.op, ast.Pow):
                operations.append("exponentiation")
            
            self.generic_visit(node)
        
        def visit_UnaryOp(self, node):
            if isinstance(node.op, ast.UAdd):
                operations.append("unary_plus")
            elif isinstance(node.op, ast.USub):
                operations.append("unary_minus")
            
            self.generic_visit(node)
        
        def visit_Call(self, node):
            # String manipulation methods
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if method_name in ['upper', 'lower', 'strip', 'lstrip', 'rstrip', 
                                  'replace', 'split', 'join', 'startswith', 'endswith',
                                  'find', 'count', 'capitalize', 'title', 'swapcase']:
                    operations.append(f"string_{method_name}")
            
            self.generic_visit(node)
        
        def visit_Subscript(self, node):
            # String slicing
            if isinstance(node.slice, ast.Slice):
                operations.append("string_slicing")
            elif isinstance(node.slice, ast.Index) or isinstance(node.slice, ast.Constant):
                operations.append("string_indexing")
            
            self.generic_visit(node)
    
    visitor = OperationVisitor()
    visitor.visit(tree)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_operations = []
    for op in operations:
        if op not in seen:
            seen.add(op)
            unique_operations.append(op)
    
    return unique_operations


def evaluate_operations(script_string):
    """
    Evaluate operations in a Python script string.
    Only processes safe operations (basic arithmetic and string concatenation).
    
    Args:
        script_string (str): The Python script as a string
        
    Returns:
        Any: The result of evaluation or an error message for unsafe operations
    """
    # Define allowed names and functions
    allowed_names = {
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Define safe operations
    safe_operations = {
        'addition', 'subtraction', 'multiplication', 'division',
        'floor_division', 'modulo', 'exponentiation', 'unary_plus',
        'unary_minus', 'string_upper', 'string_lower', 'string_strip',
        'string_lstrip', 'string_rstrip', 'string_capitalize', 'string_title',
        'string_swapcase', 'string_slicing', 'string_indexing'
    }
    
    # Parse and check operations
    operations = parse_script_operations(script_string)
    
    # Check if all operations are safe
    unsafe_ops = [op for op in operations if op not in safe_operations]
    if unsafe_ops:
        return f"Safety violation: Unsafe operations detected: {', '.join(unsafe_ops)}"
    
    try:
        tree = ast.parse(script_string)
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"
    
    # Check for potentially unsafe nodes
    class SafetyChecker(ast.NodeVisitor):
        def __init__(self):
            self.safe = True
            self.violations = []
        
        def visit_Import(self, node):
            self.safe = False
            self.violations.append("import statements")
        
        def visit_ImportFrom(self, node):
            self.safe = False
            self.violations.append("import statements")
        
        def visit_FunctionDef(self, node):
            self.safe = False
            self.violations.append("function definitions")
        
        def visit_ClassDef(self, node):
            self.safe = False
            self.violations.append("class definitions")
        
        def visit_Lambda(self, node):
            self.safe = False
            self.violations.append("lambda functions")
        
        def visit_Exec(self, node):
            self.safe = False
            self.violations.append("exec statements")
        
        def visit_Global(self, node):
            self.safe = False
            self.violations.append("global statements")
        
        def visit_Nonlocal(self, node):
            self.safe = False
            self.violations.append("nonlocal statements")
        
        def visit_Delete(self, node):
            self.safe = False
            self.violations.append("delete statements")
        
        def visit_With(self, node):
            self.safe = False
            self.violations.append("with statements")
        
        def visit_Raise(self, node):
            self.safe = False
            self.violations.append("raise statements")
        
        def visit_Try(self, node):
            self.safe = False
            self.violations.append("try statements")
        
        def visit_Assert(self, node):
            self.safe = False
            self.violations.append("assert statements")
        
        def visit_Call(self, node):
            # Check for unsafe built-in functions
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile', '__import__', 
                                   'open', 'input', 'raw_input', 'file',
                                   'execfile', 'reload', 'vars', 'locals',
                                   'globals', 'dir', 'getattr', 'setattr',
                                   'delattr', 'hasattr', 'type', 'isinstance',
                                   'issubclass', 'callable', 'classmethod',
                                   'staticmethod', 'property', 'super']:
                    self.safe = False
                    self.violations.append(f"unsafe function: {node.func.id}")
            self.generic_visit(node)
        
        def visit_Attribute(self, node):
            # Check for unsafe attributes
            if node.attr.startswith('_'):
                self.safe = False
                self.violations.append(f"private attribute access: {node.attr}")
            self.generic_visit(node)
    
    checker = SafetyChecker()
    checker.visit(tree)
    
    if not checker.safe:
        return f"Safety violation: {', '.join(set(checker.violations))}"
    
    # Safe evaluation with restricted namespace
    safe_namespace = {
        '__builtins__': {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'len': len,
            'range': range,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'pow': pow,
        }
    }
    safe_namespace.update(allowed_names)
    
    try:
        # Compile and execute
        code = compile(tree, '<string>', 'exec')
        exec(code, safe_namespace)
        
        # Try to evaluate as expression to get result
        try:
            expr_tree = ast.parse(script_string, mode='eval')
            code = compile(expr_tree, '<string>', 'eval')
            result = eval(code, safe_namespace)
            return result
        except:
            # If not an expression, check for printed output or variables
            return "Execution completed successfully"
            
    except Exception as e:
        return f"Execution error: {str(e)}"
