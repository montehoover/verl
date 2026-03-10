import ast
import logging

# Configure logger
logger = logging.getLogger(__name__)

def _get_disallowed_nodes():
    return (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.comprehension,
    )

def _get_disallowed_names():
    return {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input',
        'breakpoint', 'help', 'dir', 'locals', 'globals', 'vars',
        'setattr', 'delattr', 'getattr', 'hasattr', 'type', 'isinstance',
        'issubclass', 'super', 'property', 'classmethod', 'staticmethod',
        'callable', 'format', 'repr', 'ascii', 'ord', 'chr', 'bin',
        'hex', 'oct', 'abs', 'round', 'divmod', 'pow', 'sorted',
        'enumerate', 'filter', 'map', 'zip', 'reversed', 'iter',
        'next', 'range', 'slice', 'bytes', 'bytearray', 'memoryview',
        'complex', 'bool', 'list', 'tuple', 'set', 'frozenset', 'dict',
        'str', 'int', 'float', 'object', 'all', 'any', 'sum', 'min',
        'max', 'len', 'print', 'id', 'hash'
    }

def _get_disallowed_attributes():
    return {
        '__subclasses__', '__bases__', '__code__', 
        '__globals__', '__builtins__', '__class__', '__dict__',
        '__module__', '__name__', '__qualname__', '__annotations__'
    }

def _validate_ast_node(node, disallowed_nodes, disallowed_names, disallowed_attributes):
    if isinstance(node, disallowed_nodes):
        raise ValueError("Disallowed operation detected")
    
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in disallowed_names:
            raise ValueError("Disallowed operation detected")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in disallowed_attributes:
                raise ValueError("Disallowed operation detected")
    
    if isinstance(node, ast.Attribute):
        if node.attr in disallowed_attributes:
            raise ValueError("Disallowed operation detected")
    
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {'__builtins__', 'globals', 'locals'}:
            raise ValueError("Disallowed operation detected")

def _parse_and_validate(snippet, mode='exec'):
    disallowed_nodes = _get_disallowed_nodes()
    disallowed_names = _get_disallowed_names()
    disallowed_attributes = _get_disallowed_attributes()
    
    try:
        tree = ast.parse(snippet, mode=mode)
    except SyntaxError:
        raise ValueError("Disallowed operation detected")
    
    for node in ast.walk(tree):
        _validate_ast_node(node, disallowed_nodes, disallowed_names, disallowed_attributes)
    
    return tree

def _create_safe_namespace():
    return {
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
        }
    }

def _execute_code(tree, namespace):
    compiled = compile(tree, '<string>', 'exec')
    exec(compiled, namespace)

def _try_eval_expression(snippet, namespace):
    try:
        expr_tree = _parse_and_validate(snippet, mode='eval')
        result = eval(compile(expr_tree, '<string>', 'eval'), namespace)
        return result
    except:
        return None

def run_code_snippet(snippet: str):
    logger.info(f"Attempting to execute code snippet: {repr(snippet[:100])}{'...' if len(snippet) > 100 else ''}")
    
    try:
        tree = _parse_and_validate(snippet, mode='exec')
        logger.debug("Code snippet parsed and validated successfully")
        
        namespace = _create_safe_namespace()
        _execute_code(tree, namespace)
        logger.debug("Code snippet executed successfully")
        
        result = _try_eval_expression(snippet, namespace)
        
        if result is not None:
            logger.info(f"Code snippet execution completed with result: {repr(result)}")
        else:
            logger.info("Code snippet execution completed with no return value")
        
        return result
    except ValueError as e:
        logger.warning(f"Code snippet execution failed - {str(e)}: {repr(snippet[:100])}{'...' if len(snippet) > 100 else ''}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during code snippet execution: {repr(e)} - Snippet: {repr(snippet[:100])}{'...' if len(snippet) > 100 else ''}")
        raise ValueError("Disallowed operation detected")
