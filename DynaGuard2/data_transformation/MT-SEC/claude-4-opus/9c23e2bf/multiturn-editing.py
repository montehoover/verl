import re
import operator
from functools import partial


def extract_fields(data_list, field_names):
    """
    Extract specified fields from a list of dictionaries.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
    
    Returns:
        New list of dictionaries with only the specified fields
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
    Filter records based on conditions and extract specified fields.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        filter_conditions: Dictionary with field names as keys and 
                          callable conditions or values as values
    
    Returns:
        New list of dictionaries with only the specified fields
        from records that match all filter conditions
    """
    result = []
    for record in data_list:
        # Check if record meets all filter conditions
        meets_conditions = True
        for field, condition in filter_conditions.items():
            if field not in record:
                meets_conditions = False
                break
            
            # If condition is callable, use it as a function
            if callable(condition):
                if not condition(record[field]):
                    meets_conditions = False
                    break
            # Otherwise, check for equality
            else:
                if record[field] != condition:
                    meets_conditions = False
                    break
        
        # If record meets all conditions, extract specified fields
        if meets_conditions:
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result


def run_custom_query(dataset, query):
    """
    Execute a SQL-like query on a dataset.
    
    Args:
        dataset: List of dictionaries representing the data
        query: SQL-like query string supporting SELECT, WHERE, and ORDER BY
    
    Returns:
        List of dictionaries containing query results
    
    Raises:
        ValueError: If query is malformed or cannot be processed
    """
    # Normalize query to handle case variations
    normalized_query = ' '.join(query.split())
    
    # Parse SELECT clause
    select_match = re.match(r'SELECT\s+(.*?)\s+FROM', normalized_query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Query must contain SELECT ... FROM clause")
    
    select_fields = select_match.group(1).strip()
    if select_fields == '*':
        fields = None  # Select all fields
    else:
        fields = [f.strip() for f in select_fields.split(',')]
    
    # Start with full dataset
    result = dataset.copy()
    
    # Parse WHERE clause if present
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', normalized_query, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1).strip()
        result = apply_where_clause(result, where_clause)
    
    # Parse ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+(ASC|DESC))?$', normalized_query, re.IGNORECASE)
    if order_match:
        order_field = order_match.group(1).strip()
        order_direction = order_match.group(2)
        reverse = order_direction and order_direction.upper() == 'DESC'
        result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=reverse)
    
    # Apply field selection
    if fields is not None:
        result = extract_fields(result, fields)
    
    return result


def apply_where_clause(dataset, where_clause):
    """
    Apply WHERE clause filtering to dataset.
    
    Args:
        dataset: List of dictionaries
        where_clause: WHERE clause string
    
    Returns:
        Filtered list of dictionaries
    """
    # Parse simple conditions (field operator value)
    # Support =, !=, <, >, <=, >=
    condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)'
    match = re.match(condition_pattern, where_clause)
    
    if not match:
        raise ValueError(f"Invalid WHERE clause: {where_clause}")
    
    field, op, value = match.groups()
    field = field.strip()
    value = value.strip()
    
    # Try to parse value as number or string
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]  # Remove quotes
    else:
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Keep as string if not a number
            pass
    
    # Map operators to functions
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
    
    op_func = op_map[op]
    
    # Filter records
    filtered = []
    for record in dataset:
        if field in record:
            record_value = record[field]
            try:
                if op_func(record_value, value):
                    filtered.append(record)
            except TypeError:
                # Skip records where comparison fails due to type mismatch
                pass
    
    return filtered
