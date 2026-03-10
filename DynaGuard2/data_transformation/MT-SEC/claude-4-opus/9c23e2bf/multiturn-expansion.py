from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_fields(records, fields):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        records: List of dictionaries representing data records
        fields: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected = {}
        for field in fields:
            if field in record:
                selected[field] = record[field]
            else:
                selected[field] = None
        result.append(selected)
    
    return result

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a condition.
    
    Args:
        records: List of dictionaries representing data records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    return [record for record in records if condition(record)]

def run_custom_query(dataset: List[Dict], query: str) -> List[Dict]:
    """
    Execute SQL-like queries on the dataset.
    
    Args:
        dataset: List of dictionaries representing data records
        query: SQL-like query string
        
    Returns:
        List of dictionaries with query results
        
    Raises:
        ValueError: If query is malformed or fails
    """
    try:
        # Parse SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query, re.IGNORECASE)
        if not select_match:
            raise ValueError("Invalid query: SELECT clause not found")
        
        select_fields_str = select_match.group(1).strip()
        if select_fields_str == '*':
            selected_fields = None
        else:
            selected_fields = [field.strip() for field in select_fields_str.split(',')]
        
        # Working copy of dataset
        result = dataset.copy()
        
        # Parse WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Parse conditions (simple implementation)
            condition_pattern = r'(\w+)\s*(=|!=|>|<|>=|<=)\s*(?:\'([^\']*)\')|([\d.]+)'
            conditions = re.findall(condition_pattern, where_clause)
            
            for field, op, str_value, num_value in conditions:
                if str_value:
                    value = str_value
                elif num_value:
                    try:
                        value = float(num_value) if '.' in num_value else int(num_value)
                    except ValueError:
                        raise ValueError(f"Invalid numeric value: {num_value}")
                else:
                    raise ValueError(f"Invalid condition value")
                
                # Apply filter
                ops = {
                    '=': operator.eq,
                    '!=': operator.ne,
                    '>': operator.gt,
                    '<': operator.lt,
                    '>=': operator.ge,
                    '<=': operator.le
                }
                
                if op not in ops:
                    raise ValueError(f"Unsupported operator: {op}")
                
                result = [record for record in result if field in record and ops[op](record[field], value)]
        
        # Parse ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query, re.IGNORECASE)
        if order_match:
            order_field = order_match.group(1)
            order_direction = order_match.group(2) or 'ASC'
            
            reverse = order_direction.upper() == 'DESC'
            result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=reverse)
        
        # Apply SELECT fields
        if selected_fields is not None:
            result = select_fields(result, selected_fields)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Query failed: {str(e)}")
