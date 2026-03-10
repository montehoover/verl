import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QuizApp.QueryEvaluator')


def _is_query_safe(query: str) -> bool:
    """Check if the query contains any dangerous keywords."""
    dangerous_keywords = [
        '__import__', 'exec', 'eval', 'compile', 'open',
        'file', 'input', 'raw_input', '__builtins__',
        'globals', 'locals', 'vars', 'dir', 'getattr',
        'setattr', 'delattr', 'hasattr', '__', 'import',
        'os', 'sys', 'subprocess', 'socket', 'requests',
        'urllib', 'pickle', 'shelve', 'marshal', 'tempfile',
        'shutil', 'pathlib', 'Path', 'breakpoint', 'help',
        'quit', 'exit', 'copyright', 'credits', 'license'
    ]
    
    query_lower = query.lower()
    for keyword in dangerous_keywords:
        if keyword in query_lower:
            logger.warning(f"Dangerous keyword '{keyword}' detected in query: {query}")
            return False
    return True


def _evaluate_query(query: str) -> str:
    """Evaluate the query and return the result as a string."""
    try:
        result = eval(query, {"__builtins__": {}}, {})
        return str(result)
    except:
        raise


def run_user_query(query: str) -> str:
    start_time = datetime.datetime.now()
    
    try:
        logger.info(f"Processing query: {query}")
        
        if not _is_query_safe(query):
            logger.error(f"Query rejected due to safety concerns: {query}")
            return 'Error!'
        
        result = _evaluate_query(query)
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Query evaluated successfully: {query} -> Result: {result} (Duration: {duration:.4f}s)")
        return result
        
    except Exception as e:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"Query evaluation failed: {query} -> Error: {str(e)} (Duration: {duration:.4f}s)")
        return 'Error!'
