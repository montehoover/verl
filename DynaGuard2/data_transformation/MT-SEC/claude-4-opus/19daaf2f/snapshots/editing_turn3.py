import re
import operator
from functools import partial


def extract_fields(data_list, field_names):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    for record in data_list:
        extracted_record = {}
        for field in field_names:
            if field in record:
                extracted_record[field] = record[field]
        result.append(extracted_record)
    return result


def filter_and_extract(data_list, field_names, filter_conditions):
    """
    Filter records based on conditions and extract specific fields.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        filter_conditions: Dictionary where keys are field names and values are required values
        
    Returns:
        List of dictionaries containing only the specified fields from filtered records
    """
    result = []
    for record in data_list:
        # Check if record matches all filter conditions
        matches = True
        for field, value in filter_conditions.items():
            if field not in record or record[field] != value:
                matches = False
                break
        
        if matches:
            # Extract specified fields
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result


def run_sql_query(dataset, sql_query):
    """
    Parse and execute SQL-like queries on a dataset.
    
    Args:
        dataset: List of dictionaries representing the dataset
        sql_query: SQL-like query string
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        ValueError: If query is malformed or execution fails
    """
    # Normalize query to uppercase for parsing
    query_upper = sql_query.upper()
    
    # Parse SELECT clause
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query_upper)
    if not select_match:
        raise ValueError("Malformed query: Missing SELECT clause")
    
    select_clause = sql_query[select_match.start(1):select_match.end(1)]
    
    # Parse field names from SELECT
    if select_clause.strip() == '*':
        fields = None  # Select all fields
    else:
        fields = [field.strip() for field in select_clause.split(',')]
    
    # Parse WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.+?)(?:ORDER\s+BY|$)', query_upper)
    conditions = []
    
    if where_match:
        where_clause = sql_query[where_match.start(1):where_match.end(1)]
        
        # Parse conditions
        condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(["\']?)(.+?)\3'
        for match in re.finditer(condition_pattern, where_clause):
            field = match.group(1)
            op_str = match.group(2)
            value = match.group(4)
            
            # Convert value to appropriate type
            if match.group(3):  # Quoted string
                typed_value = value
            else:
                try:
                    typed_value = int(value)
                except ValueError:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        typed_value = value
            
            # Map operator strings to functions
            op_map = {
                '=': operator.eq,
                '!=': operator.ne,
                '<': operator.lt,
                '>': operator.gt,
                '<=': operator.le,
                '>=': operator.ge
            }
            
            if op_str not in op_map:
                raise ValueError(f"Unknown operator: {op_str}")
            
            conditions.append((field, op_map[op_str], typed_value))
    
    # Parse ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query_upper)
    order_field = None
    order_desc = False
    
    if order_match:
        order_field = sql_query[order_match.start(1):order_match.end(1)]
        if order_match.group(2) == 'DESC':
            order_desc = True
    
    # Execute query
    result = []
    
    for record in dataset:
        # Check WHERE conditions
        if conditions:
            match = True
            for field, op_func, value in conditions:
                if field not in record:
                    match = False
                    break
                if not op_func(record[field], value):
                    match = False
                    break
            if not match:
                continue
        
        # Select fields
        if fields is None:
            result.append(record.copy())
        else:
            selected = {}
            for field in fields:
                if field in record:
                    selected[field] = record[field]
            if selected:  # Only add if at least one field exists
                result.append(selected)
    
    # Apply ORDER BY
    if order_field and result:
        try:
            result.sort(key=lambda x: x.get(order_field, ''), reverse=order_desc)
        except Exception:
            raise ValueError(f"Cannot sort by field: {order_field}")
    
    return result
