import ast
import operator

# Define safe built-in functions
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'chr', 'divmod',
    'enumerate', 'filter', 'float', 'format', 'hex', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next',
    'oct', 'ord', 'pow', 'range', 'repr', 'reversed', 'round', 'set',
    'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
}

# Define safe operators
SAFE_OPERATORS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.MatMult, ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.And, ast.Or
}

# Define safe node types
SAFE_NODES = {
    ast.Module, ast.Expr, ast.Load, ast.Store, ast.Del,
    ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.While, ast.If, ast.With, ast.withitem,
    ast.Break, ast.Continue, ast.Pass,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Call, ast.keyword, ast.IfExp,
    ast.Attribute,
    ast.Subscript, ast.Slice, ast.Index,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension,
    ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Constant,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.Name, ast.Starred,
    ast.Ellipsis,
    ast.FormattedValue, ast.JoinedStr,
    ast.Return,  # Allow return statements
    ast.FunctionDef, ast.arguments, ast.arg,  # Allow function definitions
    ast.Lambda,  # Allow lambda expressions
}

class SafetyChecker(ast.NodeVisitor):
    def __init__(self):
        self.safe = True
        self.errors = []
    
    def visit_Call(self, node):
        # Check if it's a direct function call
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS:
                self.safe = False
                self.errors.append(f"Unsafe function call: {node.func.id}")
        # Check if it's a method call
        elif isinstance(node.func, ast.Attribute):
            # Allow string methods
            if isinstance(node.func.value, ast.Str) or \
               (isinstance(node.func.value, ast.Name) and node.func.attr in {
                   'append', 'extend', 'insert', 'remove', 'pop', 'clear',
                   'index', 'count', 'sort', 'reverse', 'copy',
                   'add', 'update', 'intersection', 'union', 'difference',
                   'symmetric_difference', 'issubset', 'issuperset',
                   'get', 'keys', 'values', 'items', 'popitem', 'setdefault',
                   'fromkeys', 'upper', 'lower', 'capitalize', 'title',
                   'strip', 'lstrip', 'rstrip', 'split', 'rsplit',
                   'join', 'replace', 'find', 'rfind', 'index', 'rindex',
                   'startswith', 'endswith', 'isalnum', 'isalpha', 'isdigit',
                   'isspace', 'istitle', 'isupper', 'islower', 'center',
                   'ljust', 'rjust', 'zfill', 'format', 'format_map',
                   'encode', 'decode', 'expandtabs', 'partition', 'rpartition',
                   'splitlines', 'swapcase', 'translate'
               }):
                pass  # These are safe
            else:
                self.safe = False
                self.errors.append(f"Unsafe method call: {ast.dump(node.func)}")
        else:
            self.safe = False
            self.errors.append(f"Unsafe call: {ast.dump(node)}")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.safe = False
        self.errors.append("Import statements are not allowed")
    
    def visit_ImportFrom(self, node):
        self.safe = False
        self.errors.append("Import statements are not allowed")
    
    def visit_Attribute(self, node):
        # Check for dangerous attributes
        dangerous_attrs = {'__', 'eval', 'exec', 'compile', 'open', 'input',
                          '__import__', 'globals', 'locals', 'vars', 'dir',
                          'getattr', 'setattr', 'delattr', 'hasattr',
                          '__dict__', '__class__', '__bases__', '__subclasses__',
                          '__code__', '__globals__'}
        
        if any(dangerous in node.attr for dangerous in dangerous_attrs):
            self.safe = False
            self.errors.append(f"Unsafe attribute access: {node.attr}")
        
        self.generic_visit(node)
    
    def generic_visit(self, node):
        # Check if the node type is in our safe list
        if type(node) not in SAFE_NODES:
            # Special handling for operators
            if type(node) not in SAFE_OPERATORS:
                self.safe = False
                self.errors.append(f"Unsafe node type: {type(node).__name__}")
        
        super().generic_visit(node)

def check_script_safety(script):
    """
    Check if a Python script contains only safe operations.
    
    Args:
        script (str): The Python script to check
        
    Returns:
        bool: True if the script is safe, False otherwise
    """
    try:
        # Parse the script into an AST
        tree = ast.parse(script)
        
        # Create a safety checker and visit all nodes
        checker = SafetyChecker()
        checker.visit(tree)
        
        return checker.safe
        
    except SyntaxError:
        # If the script has syntax errors, it's not safe to run
        return False
    except Exception:
        # Any other parsing errors mean it's not safe
        return False
