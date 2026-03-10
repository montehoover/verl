import re
import operator
from functools import partial

def extract_fields(dataset, fields, conditions=None):
    """
    Extract specific fields from a list of dictionaries with optional filtering.
    
    Args:
        dataset: List of dictionaries representing the dataset
        fields: List of field names to extract
        conditions: Optional dictionary of field:value pairs for filtering
        
    Returns:
        List of dictionaries containing only the specified fields that meet conditions
        
    Raises:
        ValueError: If a condition references a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if all condition fields exist in at least one record
    if conditions and dataset:
        all_fields = set()
        for record in dataset:
            all_fields.update(record.keys())
        
        for condition_field in conditions:
            if condition_field not in all_fields:
                raise ValueError(f"Condition field '{condition_field}' does not exist in dataset")
    
    result = []
    for record in dataset:
        # Check if record meets all conditions
        meets_conditions = True
        for field, value in conditions.items():
            if field not in record or record[field] != value:
                meets_conditions = False
                break
        
        if meets_conditions:
            # Extract only the requested fields
            filtered_record = {field: record.get(field) for field in fields}
            result.append(filtered_record)
    
    return result


def handle_sql_query(records, sql_command):
    """
    Parse and execute SQL-like queries on a list of dictionaries.
    
    Args:
        records: List of dictionaries representing the dataset
        sql_command: SQL-like query string
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        ValueError: If query is malformed or execution fails
    """
    # Normalize the SQL command
    sql_command = sql_command.strip()
    
    # Parse SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)\s+FROM', sql_command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Malformed query: Missing SELECT clause")
    
    select_clause = select_match.group(1).strip()
    
    # Parse fields from SELECT clause
    if select_clause == '*':
        fields = None  # Select all fields
    else:
        fields = [field.strip() for field in select_clause.split(',')]
    
    # Check for WHERE clause
    where_match = re.search(r'WHERE\s+(.+)$', sql_command, re.IGNORECASE)
    
    # Start with all records
    result = records
    
    if where_match:
        where_clause = where_match.group(1).strip()
        
        # Parse WHERE conditions
        conditions = []
        
        # Split by AND/OR
        and_parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        or_parts = []
        
        for part in and_parts:
            or_splits = re.split(r'\s+OR\s+', part, flags=re.IGNORECASE)
            if len(or_splits) > 1:
                or_parts.extend(or_splits)
            else:
                or_parts.append(part)
        
        # Parse individual conditions
        ops = {
            '=': operator.eq,
            '!=': operator.ne,
            '<>': operator.ne,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le
        }
        
        filtered_records = []
        
        for record in records:
            meets_criteria = False
            
            for condition in or_parts:
                # Parse condition
                match = re.match(r'(\w+)\s*(=|!=|<>|>=|<=|>|<)\s*(.+)', condition.strip())
                if not match:
                    raise ValueError(f"Malformed WHERE condition: {condition}")
                
                field, op_str, value_str = match.groups()
                
                # Get the operator function
                if op_str not in ops:
                    raise ValueError(f"Unsupported operator: {op_str}")
                op_func = ops[op_str]
                
                # Parse value
                value_str = value_str.strip()
                if value_str.startswith("'") and value_str.endswith("'"):
                    value = value_str[1:-1]
                elif value_str.startswith('"') and value_str.endswith('"'):
                    value = value_str[1:-1]
                else:
                    # Try to parse as number
                    try:
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                
                # Check if field exists in record
                if field not in record:
                    continue
                
                # Apply condition
                try:
                    if op_func(record[field], value):
                        meets_criteria = True
                        break
                except Exception:
                    continue
            
            if meets_criteria or not or_parts:
                filtered_records.append(record)
        
        result = filtered_records
    
    # Apply field selection
    if fields is not None:
        # Validate fields exist
        if result and fields:
            available_fields = set()
            for record in result:
                available_fields.update(record.keys())
            
            for field in fields:
                if field not in available_fields:
                    raise ValueError(f"Field '{field}' does not exist in dataset")
        
        result = [{field: record.get(field) for field in fields} for record in result]
    
    return result
