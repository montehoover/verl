from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_fields(records, fields):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        records: List of dictionaries representing records
        fields: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected_record = {}
        for field in fields:
            if field in record:
                selected_record[field] = record[field]
        result.append(selected_record)
    
    return result

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a given condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it satisfies the condition
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    result = []
    
    for record in records:
        if condition(record):
            result.append(record)
    
    return result

def execute_custom_query(data: List[Dict], query: str) -> List[Dict]:
    """
    Execute SQL-like queries with SELECT, WHERE, and ORDER BY capabilities.
    
    Args:
        data: List of dictionaries representing the data
        query: SQL-like query string
        
    Returns:
        List of dictionaries with query results
        
    Raises:
        ValueError: If query is invalid or cannot be executed
    """
    # Parse SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: missing SELECT clause")
    
    select_clause = select_match.group(1).strip()
    if select_clause == '*':
        fields = None
    else:
        fields = [field.strip() for field in select_clause.split(',')]
    
    # Parse WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    where_conditions = []
    if where_match:
        where_clause = where_match.group(1).strip()
        # Parse simple conditions (field operator value)
        condition_pattern = r'(\w+)\s*(=|!=|>|<|>=|<=)\s*([\'"]?)([^\'"]+)\3'
        conditions = re.findall(condition_pattern, where_clause)
        
        for field, op, _, value in conditions:
            # Try to convert value to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            
            # Map operators
            op_map = {
                '=': operator.eq,
                '!=': operator.ne,
                '>': operator.gt,
                '<': operator.lt,
                '>=': operator.ge,
                '<=': operator.le
            }
            
            if op not in op_map:
                raise ValueError(f"Invalid operator: {op}")
            
            where_conditions.append((field, op_map[op], value))
    
    # Parse ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC|$)', query, re.IGNORECASE)
    order_field = None
    order_desc = False
    if order_match:
        order_field = order_match.group(1).strip()
        if re.search(r'\s+DESC', query, re.IGNORECASE):
            order_desc = True
    
    # Apply filters
    result = data.copy()
    
    # Apply WHERE conditions
    for field, op_func, value in where_conditions:
        result = [record for record in result if field in record and op_func(record[field], value)]
    
    # Apply ORDER BY
    if order_field:
        try:
            result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=order_desc)
        except TypeError:
            # Handle mixed types by converting to string for sorting
            result = sorted(result, key=lambda x: str(x.get(order_field, '')), reverse=order_desc)
    
    # Apply SELECT fields
    if fields:
        result = select_fields(result, fields)
    
    return result
