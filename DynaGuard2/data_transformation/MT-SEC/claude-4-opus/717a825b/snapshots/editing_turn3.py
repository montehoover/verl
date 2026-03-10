import re
import operator
from functools import partial

def extract_fields(records, fields, filters=None):
    """
    Extract specified fields from a list of dictionaries with optional filtering.
    
    Args:
        records: List of dictionaries
        fields: List of field names to extract
        filters: Dictionary of field names to filter values/functions
                 e.g., {'age': 25, 'name': lambda x: x.startswith('A')}
        
    Returns:
        List of dictionaries containing only the specified fields that meet filter conditions
        
    Raises:
        ValueError: If a field is not found in any record
    """
    result = []
    
    for record in records:
        # Check if record meets filter conditions
        if filters:
            skip_record = False
            for filter_field, filter_value in filters.items():
                if filter_field not in record:
                    raise ValueError(f"Filter field '{filter_field}' not found in record")
                
                # If filter_value is callable, use it as a predicate function
                if callable(filter_value):
                    if not filter_value(record[filter_field]):
                        skip_record = True
                        break
                # Otherwise, check for equality
                else:
                    if record[filter_field] != filter_value:
                        skip_record = True
                        break
            
            if skip_record:
                continue
        
        # Extract specified fields
        extracted = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record")
            extracted[field] = record[field]
        result.append(extracted)
    
    return result


def execute_custom_query(data, query):
    """
    Execute a SQL-like query on a list of dictionaries.
    
    Args:
        data: List of dictionaries
        query: SQL-like query string supporting SELECT, WHERE, and ORDER BY
        
    Returns:
        List of dictionaries with query results
    """
    # Parse the query
    query = query.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: missing SELECT clause")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract WHERE clause if present
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    where_conditions = []
    if where_match:
        where_clause = where_match.group(1)
        # Parse simple conditions (field operator value)
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
                        pass
            
            # Map operator strings to operator functions
            op_map = {
                '=': operator.eq,
                '!=': operator.ne,
                '<': operator.lt,
                '>': operator.gt,
                '<=': operator.le,
                '>=': operator.ge
            }
            
            where_conditions.append((field, op_map[op], value))
    
    # Extract ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query, re.IGNORECASE)
    order_field = None
    order_desc = False
    if order_match:
        order_field = order_match.group(1)
        order_desc = order_match.group(2) and order_match.group(2).upper() == 'DESC'
    
    # Apply WHERE conditions
    filtered_data = []
    for record in data:
        include = True
        for field, op_func, value in where_conditions:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record")
            if not op_func(record[field], value):
                include = False
                break
        if include:
            filtered_data.append(record)
    
    # Apply ORDER BY
    if order_field:
        if filtered_data and order_field not in filtered_data[0]:
            raise ValueError(f"Field '{order_field}' not found in record")
        filtered_data.sort(key=lambda x: x[order_field], reverse=order_desc)
    
    # Apply SELECT
    result = []
    for record in filtered_data:
        if select_fields == ['*']:
            result.append(record.copy())
        else:
            selected = {}
            for field in select_fields:
                if field not in record:
                    raise ValueError(f"Field '{field}' not found in record")
                selected[field] = record[field]
            result.append(selected)
    
    return result
