from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_columns(records, columns):
    """
    Extract specific fields from a dataset.
    
    Args:
        records: List of dictionaries representing records
        columns: List of strings representing column names to extract
        
    Returns:
        List of dictionaries containing only the specified columns
    """
    result = []
    
    for record in records:
        selected_record = {}
        for column in columns:
            if column in record:
                selected_record[column] = record[column]
            else:
                selected_record[column] = None
        result.append(selected_record)
    
    return result


def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a given condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    return [record for record in records if condition(record)]


def handle_sql_query(records: List[Dict], sql_command: str) -> List[Dict]:
    """
    Execute SQL-like queries with SELECT, WHERE, and ORDER BY capabilities.
    
    Args:
        records: List of dictionaries representing records
        sql_command: String containing the SQL-like query
        
    Returns:
        List of dictionaries with the query results
        
    Raises:
        ValueError: If the query is malformed or fails
    """
    try:
        # Normalize the query
        sql_command = sql_command.strip()
        
        # Parse SELECT clause
        select_match = re.match(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', sql_command, re.IGNORECASE)
        if not select_match:
            raise ValueError("Invalid query: missing SELECT clause")
        
        select_clause = select_match.group(1).strip()
        
        # Parse columns
        if select_clause == '*':
            columns = None  # Select all columns
        else:
            columns = [col.strip() for col in select_clause.split(',')]
        
        # Parse WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_command, re.IGNORECASE)
        where_condition = None
        if where_match:
            where_clause = where_match.group(1).strip()
            where_condition = parse_where_clause(where_clause)
        
        # Parse ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC|$)', sql_command, re.IGNORECASE)
        order_column = None
        order_desc = False
        if order_match:
            order_column = order_match.group(1).strip()
            desc_match = re.search(r'DESC(?:\s|$)', sql_command, re.IGNORECASE)
            order_desc = bool(desc_match)
        
        # Apply WHERE filter
        result = records
        if where_condition:
            result = apply_filter(result, where_condition)
        
        # Apply ORDER BY
        if order_column:
            result = sorted(result, key=lambda x: x.get(order_column, ''), reverse=order_desc)
        
        # Apply SELECT columns
        if columns:
            result = select_columns(result, columns)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Query execution failed: {str(e)}")


def parse_where_clause(where_clause: str) -> Callable[[Dict], bool]:
    """
    Parse WHERE clause and return a filter function.
    
    Args:
        where_clause: String containing the WHERE conditions
        
    Returns:
        A callable that evaluates the WHERE condition for a record
    """
    # Handle AND/OR operations
    if ' AND ' in where_clause.upper():
        conditions = where_clause.split(' AND ')
        sub_conditions = [parse_where_clause(cond.strip()) for cond in conditions]
        return lambda record: all(cond(record) for cond in sub_conditions)
    
    if ' OR ' in where_clause.upper():
        conditions = where_clause.split(' OR ')
        sub_conditions = [parse_where_clause(cond.strip()) for cond in conditions]
        return lambda record: any(cond(record) for cond in sub_conditions)
    
    # Parse simple condition
    operators_map = {
        '>=': operator.ge,
        '<=': operator.le,
        '!=': operator.ne,
        '=': operator.eq,
        '>': operator.gt,
        '<': operator.lt,
    }
    
    for op_str, op_func in operators_map.items():
        if op_str in where_clause:
            parts = where_clause.split(op_str, 1)
            if len(parts) == 2:
                column = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                
                # Try to convert to number if possible
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                
                return lambda record, col=column, val=value, op=op_func: col in record and op(record[col], val)
    
    raise ValueError(f"Invalid WHERE clause: {where_clause}")
