import ast
import sys
import builtins

class _SafetyChecker(ast.NodeVisitor):
    def __init__(self):
        self.safe = True
        
        self.PY_38_PLUS = sys.version_info >= (3, 8)
        self.PY_36_PLUS = sys.version_info >= (3, 6)

        self.ALLOWED_NODE_TYPES = {
            ast.Module, ast.Expr,
            ast.Name, 
            ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
            ast.Assign, ast.AugAssign,
            ast.If, ast.For, ast.While,
            ast.Pass, ast.Break, ast.Continue,
            ast.List, ast.Tuple, ast.Dict, ast.Set,
            ast.Subscript, ast.Slice,
            ast.Call,
            # Comprehensions
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
            ast.comprehension,
        }

        if self.PY_38_PLUS:
            self.ALLOWED_NODE_TYPES.add(ast.Constant)
        else:
            self.ALLOWED_NODE_TYPES.update({ast.Num, ast.Str, ast.Bytes, ast.NameConstant})

        if self.PY_36_PLUS:
            self.ALLOWED_NODE_TYPES.add(ast.JoinedStr) # f-string
            self.ALLOWED_NODE_TYPES.add(ast.FormattedValue) # part of f-string
        
        self.ALLOWED_BINOP_OPERATORS = {
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
        }
        self.ALLOWED_UNARYOP_OPERATORS = {
            ast.UAdd, ast.USub, ast.Not
        }
        self.ALLOWED_BOOLOP_OPERATORS = { 
            ast.And, ast.Or
        }
        self.ALLOWED_COMPARE_OPERATORS = {
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn
        }
        self.SAFE_BUILTIN_FUNCTIONS = {
            'print', 'len', 'int', 'float', 'str', 'range', 'abs', 'round', 'min', 'max', 'sum',
            'list', 'tuple', 'dict', 'set', 'bool', 'all', 'any', 'chr', 'ord', 'divmod',
            'enumerate', 'hex', 'oct', 'pow', 'repr', 'reversed', 'slice', 'sorted', 'zip'
        }
        self.DISALLOWED_NAMES = {'eval', 'exec', 'open', '__import__', 'compile', 
                                 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars'}

    def generic_visit(self, node):
        if not self.safe:
            return

        node_type = type(node)
        if node_type not in self.ALLOWED_NODE_TYPES:
            self.safe = False
            return
        
        super().generic_visit(node)

    def visit_Name(self, node):
        if not self.safe: return
        if node.id in self.DISALLOWED_NAMES:
            self.safe = False
            return
        super().generic_visit(node)

    # For Python 3.8+
    def visit_Constant(self, node):
        if not self.safe: return
        super().generic_visit(node)

    # For Python < 3.8
    def visit_Num(self, node):
        if not self.safe: return
        super().generic_visit(node)

    def visit_Str(self, node):
        if not self.safe: return
        super().generic_visit(node)

    def visit_Bytes(self, node):
        if not self.safe: return
        super().generic_visit(node)

    def visit_NameConstant(self, node): # True, False, None
        if not self.safe: return
        super().generic_visit(node)

    # F-strings (Python 3.6+)
    def visit_JoinedStr(self, node):
        if not self.safe: return
        super().generic_visit(node)

    def visit_FormattedValue(self, node):
        if not self.safe: return
        self.visit(node.value)
        if node.format_spec:
            self.visit(node.format_spec)

    def visit_BinOp(self, node):
        if not self.safe: return
        if type(node.op) not in self.ALLOWED_BINOP_OPERATORS:
            self.safe = False
            return
        super().generic_visit(node)

    def visit_UnaryOp(self, node):
        if not self.safe: return
        if type(node.op) not in self.ALLOWED_UNARYOP_OPERATORS:
            self.safe = False
            return
        super().generic_visit(node)

    def visit_BoolOp(self, node):
        if not self.safe: return
        if type(node.op) not in self.ALLOWED_BOOLOP_OPERATORS:
            self.safe = False
            return
        super().generic_visit(node)

    def visit_Compare(self, node):
        if not self.safe: return
        for op_type in node.ops:
            if type(op_type) not in self.ALLOWED_COMPARE_OPERATORS:
                self.safe = False
                return
        super().generic_visit(node)

    def visit_Call(self, node):
        if not self.safe: return

        if not isinstance(node.func, ast.Name): 
            self.safe = False; return
        
        if node.func.id not in self.SAFE_BUILTIN_FUNCTIONS:
            self.safe = False; return
        
        self.visit(node.func) 
        if not self.safe: return

        for arg in node.args: 
            self.visit(arg)
            if not self.safe: return
        
        for kwarg in node.keywords: 
            self.visit(kwarg.value) 
            if not self.safe: return

    def visit_comprehension(self, node): 
        if not self.safe: return
        if node.is_async: 
            self.safe = False
            return
        
        self.visit(node.target)
        if not self.safe: return
        self.visit(node.iter)
        if not self.safe: return
        for if_clause in node.ifs:
            self.visit(if_clause)
            if not self.safe: return

def check_script_safety(script_string: str) -> bool:
    """
    Checks if a Python script string contains only safe operations.
    Safe operations include basic arithmetic, string concatenations,
    and calls to a restricted set of built-in functions.
    Disallowed operations include imports, file I/O, attribute access beyond
    basic literals, defining functions/classes, eval/exec, etc.
    """
    try:
        tree = ast.parse(script_string)
    except SyntaxError:
        return False 

    checker = _SafetyChecker()
    checker.visit(tree)
    return checker.safe

_EXECUTE_SAFE_SCRIPT_TEMP_CHECKER = _SafetyChecker()
_SAFE_BUILTINS_SET = frozenset(_EXECUTE_SAFE_SCRIPT_TEMP_CHECKER.SAFE_BUILTIN_FUNCTIONS)
del _EXECUTE_SAFE_SCRIPT_TEMP_CHECKER

def execute_safe_script(script_string: str):
    """
    Executes a Python script string if it's deemed safe.
    Returns the result of the last expression in the script, or None if
    the script doesn't end with an expression.
    Returns an error message string if the script is unsafe, contains syntax errors,
    or if a runtime error occurs during execution.
    """
    if not check_script_safety(script_string):
        return "Error: Script contains unsafe operations."

    safe_builtins_dict = {
        name: getattr(builtins, name) for name in _SAFE_BUILTINS_SET
        if hasattr(builtins, name) 
    }
    
    script_globals = {"__builtins__": safe_builtins_dict}
    script_locals = {} 

    try:
        tree = ast.parse(script_string)
    except SyntaxError as e:
        return f"Error: Syntax error in script - {e}"

    if not tree.body: 
        return None

    use_type_ignores_param = sys.version_info >= (3, 8)

    try:
        if isinstance(tree.body[-1], ast.Expr):
            statements_to_exec = tree.body[:-1]
            final_expr_node = tree.body[-1].value 

            if statements_to_exec:
                if use_type_ignores_param:
                    module_ast_exec = ast.Module(body=statements_to_exec, type_ignores=[])
                else:
                    module_ast_exec = ast.Module(body=statements_to_exec)
                
                code_exec = compile(module_ast_exec, "<string>", "exec")
                exec(code_exec, script_globals, script_locals)

            expr_ast_eval = ast.Expression(body=final_expr_node)
            code_eval = compile(expr_ast_eval, "<string>", "eval")
            result = eval(code_eval, script_globals, script_locals)
            return result
        else:
            if use_type_ignores_param:
                module_ast_exec = ast.Module(body=tree.body, type_ignores=[])
            else:
                module_ast_exec = ast.Module(body=tree.body)
            
            code_exec = compile(module_ast_exec, "<string>", "exec")
            exec(code_exec, script_globals, script_locals)
            return None
    except Exception as e:
        return f"Error: Runtime error in script - {type(e).__name__}: {e}"
