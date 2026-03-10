import ast
import re

def parse_script_operations(script_string):
    """
    Parse a script string and return a list of operations it contains.
    Identifies basic arithmetic and string operations while filtering out
    potentially harmful operations.
    """
    operations = []
    
    # Define allowed operation types
    allowed_ops = {
        ast.Add: 'addition',
        ast.Sub: 'subtraction',
        ast.Mult: 'multiplication',
        ast.Div: 'division',
        ast.FloorDiv: 'floor_division',
        ast.Mod: 'modulo',
        ast.Pow: 'power',
        ast.UAdd: 'unary_plus',
        ast.USub: 'unary_minus',
        ast.Eq: 'equals',
        ast.NotEq: 'not_equals',
        ast.Lt: 'less_than',
        ast.LtE: 'less_than_equal',
        ast.Gt: 'greater_than',
        ast.GtE: 'greater_than_equal',
        ast.And: 'and',
        ast.Or: 'or',
        ast.Not: 'not'
    }
    
    # Define potentially harmful patterns
    dangerous_patterns = [
        r'__[a-zA-Z_]+__',  # dunder methods
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'__import__',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'hasattr\s*\(',
        r'type\s*\(',
        r'isinstance\s*\(',
        r'issubclass\s*\(',
        r'callable\s*\(',
        r'classmethod\s*\(',
        r'staticmethod\s*\(',
        r'property\s*\(',
        r'super\s*\(',
        r'object\s*\(',
        r'__builtins__',
        r'__loader__',
        r'__package__',
        r'__spec__',
        r'__file__',
        r'__cached__',
        r'__doc__',
        r'__name__',
        r'__qualname__',
        r'__annotations__',
        r'__dict__',
        r'__weakref__',
        r'__module__',
        r'__init__',
        r'__new__',
        r'__del__',
        r'__repr__',
        r'__str__',
        r'__bytes__',
        r'__format__',
        r'__hash__',
        r'__bool__',
        r'__getattr__',
        r'__setattr__',
        r'__delattr__',
        r'__getattribute__',
        r'__setattribute__',
        r'__delattribute__',
        r'__call__',
        r'__len__',
        r'__length_hint__',
        r'__getitem__',
        r'__setitem__',
        r'__delitem__',
        r'__missing__',
        r'__iter__',
        r'__reversed__',
        r'__contains__',
        r'__add__',
        r'__sub__',
        r'__mul__',
        r'__matmul__',
        r'__truediv__',
        r'__floordiv__',
        r'__mod__',
        r'__divmod__',
        r'__pow__',
        r'__lshift__',
        r'__rshift__',
        r'__and__',
        r'__xor__',
        r'__or__',
        r'__radd__',
        r'__rsub__',
        r'__rmul__',
        r'__rmatmul__',
        r'__rtruediv__',
        r'__rfloordiv__',
        r'__rmod__',
        r'__rdivmod__',
        r'__rpow__',
        r'__rlshift__',
        r'__rrshift__',
        r'__rand__',
        r'__rxor__',
        r'__ror__',
        r'__iadd__',
        r'__isub__',
        r'__imul__',
        r'__imatmul__',
        r'__itruediv__',
        r'__ifloordiv__',
        r'__imod__',
        r'__ipow__',
        r'__ilshift__',
        r'__irshift__',
        r'__iand__',
        r'__ixor__',
        r'__ior__',
        r'__neg__',
        r'__pos__',
        r'__abs__',
        r'__invert__',
        r'__complex__',
        r'__int__',
        r'__float__',
        r'__index__',
        r'__round__',
        r'__trunc__',
        r'__floor__',
        r'__ceil__',
        r'__enter__',
        r'__exit__',
        r'__await__',
        r'__aiter__',
        r'__anext__',
        r'__aenter__',
        r'__aexit__',
        r'os\.',
        r'sys\.',
        r'subprocess',
        r'importlib',
        r'pkgutil',
        r'inspect',
        r'ast\.',
        r'code\.',
        r'codeop\.',
        r'compileall',
        r'dis\.',
        r'pickletools',
        r'tabnanny',
        r'tokenize',
        r'keyword\.',
        r'symbol\.',
        r'token\.',
        r'parser\.',
        r'symtable',
        r'pyclbr',
        r'py_compile',
        r'zipimport',
        r'runpy',
        r'builtins\.',
        r'warnings\.',
        r'contextlib',
        r'abc\.',
        r'atexit',
        r'traceback',
        r'__future__',
        r'gc\.',
        r'site\.',
        r'sysconfig',
        r'import\s+',
        r'from\s+.*\s+import'
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, script_string, re.IGNORECASE):
            raise ValueError(f"Potentially harmful operation detected: {pattern}")
    
    try:
        tree = ast.parse(script_string)
        
        class OperationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.operations = []
            
            def visit_BinOp(self, node):
                op_type = type(node.op)
                if op_type in allowed_ops:
                    self.operations.append({
                        'type': 'binary_operation',
                        'operation': allowed_ops[op_type],
                        'line': node.lineno if hasattr(node, 'lineno') else None
                    })
                self.generic_visit(node)
            
            def visit_UnaryOp(self, node):
                op_type = type(node.op)
                if op_type in allowed_ops:
                    self.operations.append({
                        'type': 'unary_operation',
                        'operation': allowed_ops[op_type],
                        'line': node.lineno if hasattr(node, 'lineno') else None
                    })
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                for op in node.ops:
                    op_type = type(op)
                    if op_type in allowed_ops:
                        self.operations.append({
                            'type': 'comparison',
                            'operation': allowed_ops[op_type],
                            'line': node.lineno if hasattr(node, 'lineno') else None
                        })
                self.generic_visit(node)
            
            def visit_BoolOp(self, node):
                op_type = type(node.op)
                if op_type in allowed_ops:
                    self.operations.append({
                        'type': 'boolean_operation',
                        'operation': allowed_ops[op_type],
                        'line': node.lineno if hasattr(node, 'lineno') else None
                    })
                self.generic_visit(node)
            
            def visit_Str(self, node):
                self.operations.append({
                    'type': 'string_literal',
                    'operation': 'string_creation',
                    'line': node.lineno if hasattr(node, 'lineno') else None
                })
                self.generic_visit(node)
            
            def visit_JoinedStr(self, node):
                self.operations.append({
                    'type': 'string_operation',
                    'operation': 'string_formatting',
                    'line': node.lineno if hasattr(node, 'lineno') else None
                })
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Only allow safe string operations
                    if func_name in ['str', 'int', 'float', 'len', 'abs', 'min', 'max', 'sum']:
                        self.operations.append({
                            'type': 'function_call',
                            'operation': f'{func_name}_function',
                            'line': node.lineno if hasattr(node, 'lineno') else None
                        })
                    elif func_name in ['print']:
                        self.operations.append({
                            'type': 'output_operation',
                            'operation': 'print',
                            'line': node.lineno if hasattr(node, 'lineno') else None
                        })
                elif isinstance(node.func, ast.Attribute):
                    # Check for string methods
                    if isinstance(node.func.value, ast.Str) or (isinstance(node.func.value, ast.Name)):
                        method_name = node.func.attr
                        if method_name in ['upper', 'lower', 'strip', 'lstrip', 'rstrip', 
                                         'split', 'join', 'replace', 'find', 'count',
                                         'startswith', 'endswith', 'isdigit', 'isalpha',
                                         'isalnum', 'isspace', 'isupper', 'islower']:
                            self.operations.append({
                                'type': 'string_method',
                                'operation': f'string_{method_name}',
                                'line': node.lineno if hasattr(node, 'lineno') else None
                            })
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                self.operations.append({
                    'type': 'assignment',
                    'operation': 'variable_assignment',
                    'line': node.lineno if hasattr(node, 'lineno') else None
                })
                self.generic_visit(node)
            
            def visit_AugAssign(self, node):
                op_type = type(node.op)
                if op_type in allowed_ops:
                    self.operations.append({
                        'type': 'augmented_assignment',
                        'operation': f'{allowed_ops[op_type]}_assign',
                        'line': node.lineno if hasattr(node, 'lineno') else None
                    })
                self.generic_visit(node)
        
        visitor = OperationVisitor()
        visitor.visit(tree)
        operations = visitor.operations
        
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    return operations


def execute_operations(script_string):
    """
    Execute operations from a script string safely.
    Returns the result of operations or an error message if unsafe.
    """
    try:
        # First parse to check for safety
        operations = parse_script_operations(script_string)
        
        # Define safe builtins
        safe_builtins = {
            'abs': abs,
            'int': int,
            'float': float,
            'str': str,
            'len': len,
            'min': min,
            'max': max,
            'sum': sum,
            'print': print,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Create restricted globals
        restricted_globals = {'__builtins__': safe_builtins}
        
        # Compile and execute
        try:
            compiled_code = compile(script_string, '<string>', 'exec')
            
            # Create a custom namespace to capture results
            namespace = {}
            
            # Execute in restricted environment
            exec(compiled_code, restricted_globals, namespace)
            
            # Return namespace with results
            return {
                'success': True,
                'operations': operations,
                'namespace': namespace,
                'result': namespace.get('result', None) if 'result' in namespace else 'Execution completed successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution error: {str(e)}',
                'operations': operations
            }
            
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'operations': []
        }
