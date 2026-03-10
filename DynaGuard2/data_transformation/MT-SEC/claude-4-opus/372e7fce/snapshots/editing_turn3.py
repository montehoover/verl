import re
import operator
from functools import partial

def extract_fields(data_list, field_names, conditions=None):
    """
    Extract specific fields from a list of dictionaries with optional filtering.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        conditions: Dictionary of field:value pairs for filtering (optional)
    
    Returns:
        List of dictionaries containing only the specified fields that meet all conditions
    
    Raises:
        ValueError: If a condition is based on a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if all condition fields exist in at least one record
    if conditions and data_list:
        all_fields = set()
        for record in data_list:
            all_fields.update(record.keys())
        
        for field in conditions:
            if field not in all_fields:
                raise ValueError(f"Condition field '{field}' does not exist in any record")
    
    result = []
    for record in data_list:
        # Check if record meets all conditions
        meets_conditions = True
        for field, value in conditions.items():
            if field not in record:
                raise ValueError(f"Condition field '{field}' does not exist in record")
            if record[field] != value:
                meets_conditions = False
                break
        
        if meets_conditions:
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result


def execute_query_cmd(dataset_list, sql_query):
    """
    Execute a SQL-like query on a list of dictionaries.
    
    Args:
        dataset_list: List of dictionaries representing the dataset
        sql_query: SQL-like query string (supports SELECT, WHERE, ORDER BY)
    
    Returns:
        List of dictionaries containing query results
    
    Raises:
        ValueError: If query is malformed or cannot be processed
    """
    if not sql_query:
        raise ValueError("Query cannot be empty")
    
    # Normalize query to uppercase for parsing
    query_upper = sql_query.upper()
    
    # Parse SELECT clause
    select_match = re.search(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    if not select_match:
        raise ValueError("Query must start with SELECT")
    
    select_clause = sql_query[select_match.start(1):select_match.end(1)]
    
    # Parse selected fields
    if select_clause.strip() == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    # Parse WHERE clause
    where_conditions = []
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query_upper, re.IGNORECASE)
    if where_match:
        where_clause = sql_query[where_match.start(1):where_match.end(1)]
        
        # Parse conditions (supports =, !=, <, >, <=, >=)
        condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*([\'"]?)(.+?)\3'
        conditions = re.findall(condition_pattern, where_clause)
        
        for field, op, quote, value in conditions:
            # Convert value to appropriate type
            if not quote:  # No quotes, try to convert to number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            
            # Map operator strings to operator functions
            op_map = {
                '=': operator.eq,
                '!=': operator.ne,
                '<': operator.lt,
                '>': operator.gt,
                '<=': operator.le,
                '>=': operator.ge
            }
            
            if op not in op_map:
                raise ValueError(f"Unsupported operator: {op}")
            
            where_conditions.append((field, op_map[op], value))
    
    # Parse ORDER BY clause
    order_field = None
    order_desc = False
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query_upper, re.IGNORECASE)
    if order_match:
        order_field = sql_query[order_match.start(1):order_match.end(1)]
        if order_match.group(2) and order_match.group(2).upper() == 'DESC':
            order_desc = True
    
    # Execute query
    result = []
    
    # Filter records based on WHERE conditions
    for record in dataset_list:
        include_record = True
        
        for field, op_func, value in where_conditions:
            if field not in record:
                raise ValueError(f"Field '{field}' does not exist in dataset")
            
            try:
                if not op_func(record[field], value):
                    include_record = False
                    break
            except Exception as e:
                raise ValueError(f"Cannot compare field '{field}' with value '{value}': {str(e)}")
        
        if include_record:
            # Select specified fields
            if selected_fields is None:
                result.append(record.copy())
            else:
                selected_record = {}
                for field in selected_fields:
                    if field not in record:
                        raise ValueError(f"Selected field '{field}' does not exist in dataset")
                    selected_record[field] = record[field]
                result.append(selected_record)
    
    # Sort results if ORDER BY is specified
    if order_field:
        if result and order_field not in result[0]:
            raise ValueError(f"ORDER BY field '{order_field}' does not exist in dataset")
        
        try:
            result.sort(key=lambda x: x[order_field], reverse=order_desc)
        except Exception as e:
            raise ValueError(f"Cannot sort by field '{order_field}': {str(e)}")
    
    return result
