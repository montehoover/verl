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
