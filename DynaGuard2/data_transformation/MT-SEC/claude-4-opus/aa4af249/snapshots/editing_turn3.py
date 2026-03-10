import re
import operator
from functools import partial

def check_fields(dataset, field_names, conditions=None):
    """
    Check if all specified field names exist in the dataset and optionally if any records match conditions.
    
    Args:
        dataset: List of dictionaries representing the dataset
        field_names: List of field names to check
        conditions: Optional dictionary specifying field-value pairs to match
        
    Returns:
        Boolean indicating whether all fields exist and conditions are met (if provided)
    """
    if not dataset or not field_names:
        return False
    
    # Collect all unique keys from the dataset
    all_keys = set()
    for record in dataset:
        all_keys.update(record.keys())
    
    # Check if all requested fields exist
    fields_exist = all(field in all_keys for field in field_names)
    
    if not fields_exist:
        return False
    
    # If no conditions specified, return True since fields exist
    if conditions is None:
        return True
    
    # Check if any record matches all conditions
    for record in dataset:
        if all(record.get(field) == value for field, value in conditions.items()):
            return True
    
    return False


def process_sql_request(dataset_records, sql_statement):
    """
    Parse and execute SQL-like queries on dataset records.
    
    Args:
        dataset_records: List of dictionaries representing the dataset
        sql_statement: String containing SQL-like query
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        ValueError: If query is malformed or execution fails
    """
    if not sql_statement or not isinstance(sql_statement, str):
        raise ValueError("Invalid SQL statement")
    
    # Normalize the SQL statement
    sql_upper = sql_statement.upper().strip()
    
    # Parse SELECT query
    select_pattern = r'SELECT\s+(.*?)\s+FROM\s+\w+(?:\s+WHERE\s+(.*))?'
    match = re.match(select_pattern, sql_upper, re.IGNORECASE | re.DOTALL)
    
    if not match:
        raise ValueError("Malformed SQL query - only SELECT statements are supported")
    
    # Extract fields and where clause
    fields_str = match.group(1).strip()
    where_clause = match.group(2).strip() if match.group(2) else None
    
    # Parse fields
    if fields_str == '*':
        fields = None  # Select all fields
    else:
        # Extract field names from original statement to preserve case
        fields_match = re.match(r'SELECT\s+(.*?)\s+FROM', sql_statement, re.IGNORECASE)
        if fields_match:
            fields = [f.strip() for f in fields_match.group(1).split(',')]
        else:
            raise ValueError("Failed to parse field names")
    
    # Start with all records
    result = dataset_records.copy()
    
    # Apply WHERE clause if present
    if where_clause:
        # Extract WHERE clause from original statement to preserve case
        where_match = re.search(r'WHERE\s+(.*)', sql_statement, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_original = where_match.group(1).strip()
            result = apply_where_clause(result, where_original)
        else:
            raise ValueError("Failed to parse WHERE clause")
    
    # Project fields if specific fields requested
    if fields is not None:
        projected = []
        for record in result:
            new_record = {}
            for field in fields:
                if field not in record:
                    raise ValueError(f"Field '{field}' not found in dataset")
                new_record[field] = record[field]
            projected.append(new_record)
        result = projected
    
    return result


def apply_where_clause(records, where_clause):
    """
    Apply WHERE clause filtering to records.
    
    Args:
        records: List of dictionaries to filter
        where_clause: String containing WHERE conditions
        
    Returns:
        Filtered list of records
    """
    # Define comparison operators
    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le
    }
    
    # Handle AND/OR operations
    if ' AND ' in where_clause.upper():
        conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        return filter_records_and(records, conditions, ops)
    elif ' OR ' in where_clause.upper():
        conditions = re.split(r'\s+OR\s+', where_clause, flags=re.IGNORECASE)
        return filter_records_or(records, conditions, ops)
    else:
        # Single condition
        return filter_by_condition(records, where_clause, ops)


def filter_records_and(records, conditions, ops):
    """Filter records where all conditions must be true."""
    result = records
    for condition in conditions:
        result = filter_by_condition(result, condition, ops)
    return result


def filter_records_or(records, conditions, ops):
    """Filter records where any condition can be true."""
    result_set = set()
    for condition in conditions:
        filtered = filter_by_condition(records, condition, ops)
        for record in filtered:
            # Convert dict to tuple of items for hashability
            record_tuple = tuple(sorted(record.items()))
            result_set.add(record_tuple)
    
    # Convert back to list of dicts
    return [dict(record_tuple) for record_tuple in result_set]


def filter_by_condition(records, condition, ops):
    """Filter records by a single condition."""
    # Pattern to match field operator value
    pattern = r'(\w+)\s*(=|!=|<>|>=|<=|>|<)\s*(.+)'
    match = re.match(pattern, condition.strip())
    
    if not match:
        raise ValueError(f"Invalid WHERE condition: {condition}")
    
    field = match.group(1)
    op_str = match.group(2)
    value_str = match.group(3).strip()
    
    # Parse value
    value = parse_value(value_str)
    
    # Get operator function
    op_func = ops.get(op_str)
    if not op_func:
        raise ValueError(f"Unsupported operator: {op_str}")
    
    # Filter records
    filtered = []
    for record in records:
        if field not in record:
            raise ValueError(f"Field '{field}' not found in dataset")
        
        try:
            if op_func(record[field], value):
                filtered.append(record)
        except TypeError:
            # Type mismatch in comparison
            continue
    
    return filtered


def parse_value(value_str):
    """Parse a value string into appropriate Python type."""
    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    # Try to parse as number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Return as string
        return value_str
