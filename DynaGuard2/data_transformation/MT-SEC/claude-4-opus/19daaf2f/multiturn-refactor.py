import re
import operator
from functools import partial
import logging
from datetime import datetime

def run_sql_query(dataset, sql_query):
    """
    Process a custom SQL-like query on data represented as a list of dictionaries.
    
    Args:
        dataset: A list of dictionaries where each dictionary represents a record
        sql_query: A string containing the SQL-like query to be executed
        
    Returns:
        A list containing dictionaries that represent the results of the executed query
        
    Raises:
        ValueError: Raised if the query is not correctly formed or execution fails
    """
    # Initialize logger
    logger = logging.getLogger('sql_query_processor')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Log query start
    query_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    logger.info(f"[Query {query_id}] Starting query execution")
    logger.info(f"[Query {query_id}] Raw query: {sql_query}")
    logger.info(f"[Query {query_id}] Dataset size: {len(dataset)} records")
    
    if not dataset:
        logger.info(f"[Query {query_id}] Empty dataset provided, returning empty result")
        return []
    
    try:
        # Parse the query into components
        logger.info(f"[Query {query_id}] Parsing query...")
        query_components = parse_query(sql_query)
        logger.info(f"[Query {query_id}] Query components parsed: SELECT={query_components['select']}, WHERE={query_components['where']}, ORDER BY={query_components['order_by']}")
        
        # Execute the query through a pipeline
        logger.info(f"[Query {query_id}] Building execution pipeline...")
        result = execute_pipeline(dataset, query_components, query_id, logger)
        
        # Log final results
        logger.info(f"[Query {query_id}] Query execution completed successfully")
        logger.info(f"[Query {query_id}] Result size: {len(result)} records")
        if result and len(result) <= 5:
            logger.info(f"[Query {query_id}] Sample results: {result}")
        elif result:
            logger.info(f"[Query {query_id}] Sample results (first 5): {result[:5]}")
        
        return result
        
    except Exception as e:
        logger.error(f"[Query {query_id}] Query execution failed: {str(e)}")
        raise


def parse_query(sql_query):
    """
    Parse SQL query string into structured components.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Dictionary with parsed query components
        
    Raises:
        ValueError: If query is malformed
    """
    query_upper = sql_query.upper()
    
    # Extract SELECT clause
    select_match = re.search(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_clause = sql_query[select_match.start(1):select_match.end(1)].strip()
    
    # Extract WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    where_clause = None
    if where_match:
        where_clause = sql_query[where_match.start(1):where_match.end(1)].strip()
    
    # Extract ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(.+?)$', query_upper, re.IGNORECASE)
    order_clause = None
    if order_match:
        order_clause = sql_query[order_match.start(1):order_match.end(1)].strip()
    
    # Parse SELECT fields
    if select_clause.strip() == '*':
        selected_fields = None
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    return {
        'select': selected_fields,
        'where': where_clause,
        'order_by': order_clause
    }


def execute_pipeline(dataset, query_components, query_id, logger):
    """
    Execute query components as a pipeline of transformations.
    
    Args:
        dataset: Initial dataset
        query_components: Parsed query components
        query_id: Unique query identifier for logging
        logger: Logger instance
        
    Returns:
        Transformed dataset
    """
    # Build pipeline of operations
    pipeline = []
    
    # Add WHERE filter if present
    if query_components['where']:
        logger.info(f"[Query {query_id}] Adding WHERE filter to pipeline: {query_components['where']}")
        pipeline.append(('WHERE', create_where_filter(query_components['where'])))
    
    # Add SELECT projection if specific fields requested
    if query_components['select'] is not None:
        fields_str = ', '.join(query_components['select'])
        logger.info(f"[Query {query_id}] Adding SELECT projection to pipeline: {fields_str}")
        pipeline.append(('SELECT', create_select_projection(query_components['select'])))
    
    # Add ORDER BY sort if present
    if query_components['order_by']:
        logger.info(f"[Query {query_id}] Adding ORDER BY sort to pipeline: {query_components['order_by']}")
        pipeline.append(('ORDER BY', create_order_by_sort(query_components['order_by'])))
    
    # Execute pipeline
    result = dataset
    for operation_name, operation in pipeline:
        logger.info(f"[Query {query_id}] Executing {operation_name} operation...")
        before_count = len(result)
        result = operation(result)
        after_count = len(result)
        logger.info(f"[Query {query_id}] {operation_name} completed: {before_count} → {after_count} records")
    
    return result


def create_where_filter(where_clause):
    """
    Create a filter function for WHERE clause.
    
    Args:
        where_clause: WHERE clause string
        
    Returns:
        Filter function that takes a dataset and returns filtered dataset
    """
    conditions = parse_where_conditions(where_clause)
    
    def filter_function(data):
        return [record for record in data if evaluate_conditions(record, conditions)]
    
    return filter_function


def create_select_projection(selected_fields):
    """
    Create a projection function for SELECT clause.
    
    Args:
        selected_fields: List of fields to select
        
    Returns:
        Projection function that takes a dataset and returns projected dataset
    """
    def projection_function(data):
        result = []
        for record in data:
            new_record = {}
            for field in selected_fields:
                if field in record:
                    new_record[field] = record[field]
                else:
                    raise ValueError(f"Field '{field}' not found in dataset")
            result.append(new_record)
        return result
    
    return projection_function


def create_order_by_sort(order_clause):
    """
    Create a sort function for ORDER BY clause.
    
    Args:
        order_clause: ORDER BY clause string
        
    Returns:
        Sort function that takes a dataset and returns sorted dataset
    """
    # Parse ORDER BY clause
    parts = order_clause.split(',')
    sort_keys = []
    
    for part in parts:
        part = part.strip()
        if ' DESC' in part.upper():
            field = part[:part.upper().rfind(' DESC')].strip()
            reverse = True
        elif ' ASC' in part.upper():
            field = part[:part.upper().rfind(' ASC')].strip()
            reverse = False
        else:
            field = part
            reverse = False
        
        sort_keys.append((field, reverse))
    
    def sort_function(data):
        result = data.copy()
        for field, reverse in reversed(sort_keys):
            try:
                result.sort(key=lambda x: x.get(field, ''), reverse=reverse)
            except Exception as e:
                raise ValueError(f"Error sorting by field '{field}': {str(e)}")
        return result
    
    return sort_function


def parse_where_conditions(where_clause):
    """Parse WHERE clause into a list of conditions."""
    conditions = []
    
    # Replace AND/OR with markers for splitting
    clause = where_clause.replace(' AND ', ' __AND__ ').replace(' OR ', ' __OR__ ')
    clause = clause.replace(' and ', ' __AND__ ').replace(' or ', ' __OR__ ')
    
    parts = re.split(r'\s+__(?:AND|OR)__\s+', clause)
    
    for i, part in enumerate(parts):
        # Parse individual condition
        match = re.match(r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)', part.strip())
        if not match:
            raise ValueError(f"Invalid WHERE condition: {part}")
        
        field, op, value = match.groups()
        value = value.strip().strip("'\"")
        
        # Try to convert to number if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        # Determine logical operator for this condition
        if i > 0:
            # Check what came before this condition
            before_idx = where_clause.upper().find(parts[i-1].upper()) + len(parts[i-1])
            after_idx = where_clause.upper().find(part.upper())
            between = where_clause[before_idx:after_idx].upper()
            
            if ' AND ' in between:
                logical_op = 'AND'
            else:
                logical_op = 'OR'
        else:
            logical_op = None
        
        conditions.append({
            'field': field,
            'operator': op,
            'value': value,
            'logical_op': logical_op
        })
    
    return conditions


def evaluate_conditions(record, conditions):
    """Evaluate conditions against a record."""
    if not conditions:
        return True
    
    # Map string operators to functions
    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    result = None
    
    for i, cond in enumerate(conditions):
        field_value = record.get(cond['field'])
        if field_value is None:
            cond_result = False
        else:
            try:
                cond_result = ops[cond['operator']](field_value, cond['value'])
            except:
                cond_result = False
        
        if i == 0:
            result = cond_result
        elif cond['logical_op'] == 'AND':
            result = result and cond_result
        elif cond['logical_op'] == 'OR':
            result = result or cond_result
    
    return result
