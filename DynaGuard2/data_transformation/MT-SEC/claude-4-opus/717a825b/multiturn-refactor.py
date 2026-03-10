import re
import operator
from functools import partial
import logging
import json
from datetime import datetime

def execute_custom_query(data, query):
    """
    Execute a custom SQL-like query on a list of dictionaries.
    
    Args:
        data: list of dictionaries representing the dataset
        query: str, a SQL-like query string
        
    Returns:
        list of dictionaries representing the query results
        
    Raises:
        ValueError: Raised when the query is invalid or cannot be executed
    """
    # Initialize logger
    logger = logging.getLogger('custom_query_executor')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Log query start
    query_start_time = datetime.now()
    logger.info(f"Executing query: {query}")
    logger.info(f"Input data size: {len(data)} rows")
    
    try:
        # Validate inputs
        validate_inputs(data, query)
        
        # Parse the query into components
        parsed_query = parse_query(query)
        logger.info(f"Parsed query components: SELECT={parsed_query['select']}, WHERE={parsed_query['where']}, ORDER BY={parsed_query['order_by']}")
        
        # Build and execute the pipeline
        pipeline = build_pipeline(parsed_query)
        logger.info(f"Built pipeline with {len(pipeline)} operations")
        
        result = execute_pipeline(data, pipeline, logger)
        
        # Log query completion
        query_end_time = datetime.now()
        execution_time = (query_end_time - query_start_time).total_seconds()
        logger.info(f"Query completed successfully in {execution_time:.3f} seconds")
        logger.info(f"Result size: {len(result)} rows")
        
        # Log sample of results if not too large
        if result and len(result) <= 5:
            logger.info(f"Complete results: {json.dumps(result, indent=2)}")
        elif result:
            logger.info(f"First 3 results: {json.dumps(result[:3], indent=2)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        logger.error(f"Failed query: {query}")
        raise


def validate_inputs(data, query):
    """Validate the input data and query"""
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("Data must be a list of dictionaries")
    
    if not isinstance(query, str):
        raise ValueError("Query must be a string")


def parse_query(query):
    """Parse SQL-like query into components"""
    query = query.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_clause = select_match.group(1).strip()
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.*?)$', query, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    return {
        'select': select_clause,
        'where': where_clause,
        'order_by': order_clause
    }


def build_pipeline(parsed_query):
    """Build a pipeline of operations based on the parsed query"""
    pipeline = []
    
    # Add WHERE filter if present
    if parsed_query['where']:
        pipeline.append(partial(filter_where, where_clause=parsed_query['where']))
    
    # Add SELECT projection
    pipeline.append(partial(project_select, select_clause=parsed_query['select']))
    
    # Add ORDER BY sort if present
    if parsed_query['order_by']:
        pipeline.append(partial(sort_order_by, order_clause=parsed_query['order_by']))
    
    return pipeline


def execute_pipeline(data, pipeline, logger=None):
    """Execute a pipeline of operations on the data"""
    result = data
    for i, operation in enumerate(pipeline):
        if logger:
            operation_name = operation.func.__name__
            logger.debug(f"Executing pipeline step {i+1}: {operation_name}")
            pre_size = len(result)
        
        result = operation(result)
        
        if logger:
            post_size = len(result)
            logger.debug(f"Step {i+1} complete: {pre_size} rows -> {post_size} rows")
    
    return result


def filter_where(data, where_clause):
    """Filter data based on WHERE clause"""
    filtered_data = []
    for row in data:
        if evaluate_where_clause(row, where_clause):
            filtered_data.append(row)
    return filtered_data


def project_select(data, select_clause):
    """Project data based on SELECT clause"""
    if select_clause == '*':
        return [dict(row) for row in data]
    
    selected_fields = [field.strip() for field in select_clause.split(',')]
    
    # Validate field names if data is not empty
    if data:
        all_fields = set()
        for row in data:
            all_fields.update(row.keys())
        for field in selected_fields:
            if field not in all_fields:
                raise ValueError(f"Invalid field in SELECT: {field}")
    
    result_data = []
    for row in data:
        new_row = {}
        for field in selected_fields:
            if field in row:
                new_row[field] = row[field]
        result_data.append(new_row)
    
    return result_data


def sort_order_by(data, order_clause):
    """Sort data based on ORDER BY clause"""
    parts = order_clause.split()
    if not parts:
        raise ValueError("Invalid ORDER BY clause")
    
    field = parts[0]
    ascending = True
    
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            ascending = False
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Check if field exists
    if data and field not in data[0]:
        raise ValueError(f"Invalid field in ORDER BY: {field}")
    
    # Sort data
    return sorted(data, key=lambda x: x.get(field, ''), reverse=not ascending)


def evaluate_where_clause(row, where_clause):
    """Evaluate WHERE clause for a single row"""
    # Parse conditions (supports AND only for simplicity)
    conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
    
    for condition in conditions:
        # Parse single condition
        match = re.match(r'(\w+)\s*([><=!]+)\s*(.+)', condition.strip())
        if not match:
            raise ValueError(f"Invalid WHERE condition: {condition}")
        
        field, op, value_str = match.groups()
        
        if field not in row:
            return False
        
        # Parse value
        value = parse_value(value_str.strip())
        row_value = row[field]
        
        # Apply operator
        if op == '=':
            if not (row_value == value):
                return False
        elif op == '!=':
            if not (row_value != value):
                return False
        elif op == '>':
            if not (row_value > value):
                return False
        elif op == '<':
            if not (row_value < value):
                return False
        elif op == '>=':
            if not (row_value >= value):
                return False
        elif op == '<=':
            if not (row_value <= value):
                return False
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    return True


def parse_value(value_str):
    """Parse a value from string format"""
    # Remove quotes if present
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    
    # Try to parse as number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return value_str
