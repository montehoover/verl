import re
import operator
from functools import partial
import logging
import json
from datetime import datetime

def parse_query(command):
    """Parse SQL-like command into structured components."""
    command = command.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query format: missing SELECT clause")
    
    select_clause = select_match.group(1).strip()
    
    # Extract WHERE clause if present
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', command, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+ASC|\s+DESC|$)', command, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Check if ASC or DESC is specified
    desc_match = re.search(r'ORDER\s+BY\s+.+?\s+DESC', command, re.IGNORECASE)
    ascending = not bool(desc_match)
    
    # Process SELECT clause
    if select_clause == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    return {
        'select': selected_fields,
        'where': where_clause,
        'order_by': order_clause,
        'ascending': ascending
    }

def parse_condition(condition_str):
    """Parse a single WHERE condition into components."""
    match = re.match(r'(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)', condition_str.strip())
    if not match:
        raise ValueError(f"Invalid WHERE condition: {condition_str}")
    
    field, op, value = match.groups()
    field = field.strip()
    value = value.strip()
    
    # Remove quotes if present
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    else:
        # Try to convert to number
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
    
    return field, op, value

def evaluate_condition(record, field, op, value):
    """Evaluate a single condition against a record."""
    if field not in record:
        raise ValueError(f"Field '{field}' not found in record")
    
    record_value = record[field]
    
    # Evaluate condition
    if op == '=':
        return record_value == value
    elif op == '!=' or op == '<>':
        return record_value != value
    elif op == '<':
        return record_value < value
    elif op == '>':
        return record_value > value
    elif op == '<=':
        return record_value <= value
    elif op == '>=':
        return record_value >= value
    else:
        raise ValueError(f"Unknown operator: {op}")

def evaluate_where_clause(record, where_clause):
    """Evaluate WHERE clause against a record."""
    # Handle AND conditions
    and_conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
    
    for condition in and_conditions:
        # Handle OR conditions within each AND group
        or_conditions = re.split(r'\s+OR\s+', condition, flags=re.IGNORECASE)
        or_result = False
        
        for or_condition in or_conditions:
            # Parse and evaluate individual condition
            field, op, value = parse_condition(or_condition)
            result = evaluate_condition(record, field, op, value)
            
            if result:
                or_result = True
                break
        
        if not or_result:
            return False
    
    return True

def filter_records(records, where_clause):
    """Apply WHERE clause filtering to records."""
    if not where_clause:
        return records
    
    return [record for record in records if evaluate_where_clause(record, where_clause)]

def sort_records(records, order_field, ascending):
    """Apply ORDER BY sorting to records."""
    if not order_field:
        return records
    
    try:
        return sorted(records, 
                     key=lambda x: x.get(order_field, ''), 
                     reverse=not ascending)
    except Exception:
        raise ValueError(f"Cannot order by field: {order_field}")

def select_fields(records, selected_fields):
    """Apply SELECT clause projection to records."""
    if selected_fields is None:
        return [record.copy() for record in records]
    
    result = []
    for record in records:
        selected_record = {}
        for field in selected_fields:
            if field in record:
                selected_record[field] = record[field]
            else:
                raise ValueError(f"Field '{field}' not found in records")
        result.append(selected_record)
    
    return result

def setup_query_logger():
    """Setup logger for SQL queries."""
    logger = logging.getLogger('sql_query_logger')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create console handler with formatting
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '\n%(asctime)s - SQL Query Execution\n%(message)s\n' + '-' * 80,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def format_query_log(command, input_count, output_count, duration_ms, error=None):
    """Format query information for logging."""
    log_parts = []
    log_parts.append(f"Query: {command}")
    log_parts.append(f"Input Records: {input_count}")
    
    if error:
        log_parts.append(f"Status: FAILED")
        log_parts.append(f"Error: {error}")
    else:
        log_parts.append(f"Status: SUCCESS")
        log_parts.append(f"Output Records: {output_count}")
        log_parts.append(f"Execution Time: {duration_ms:.2f}ms")
    
    return '\n'.join(log_parts)

def run_sql_query(records, command):
    """Execute a SQL-like query on records using a pipeline pattern."""
    # Setup logging
    logger = setup_query_logger()
    
    # Track execution time
    start_time = datetime.now()
    input_count = len(records)
    
    try:
        # Parse the query
        query_parts = parse_query(command)
        
        # Build and execute the pipeline
        result = records
        result = filter_records(result, query_parts['where'])
        result = sort_records(result, query_parts['order_by'], query_parts['ascending'])
        result = select_fields(result, query_parts['select'])
        
        # Calculate execution time
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Log successful query
        output_count = len(result)
        log_message = format_query_log(command, input_count, output_count, duration_ms)
        logger.info(log_message)
        
        return result
        
    except Exception as e:
        # Calculate execution time
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Log failed query
        log_message = format_query_log(command, input_count, 0, duration_ms, error=str(e))
        logger.error(log_message)
        
        # Re-raise the exception
        raise
