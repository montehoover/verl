import re
import operator
from functools import partial

def extract_fields(data_list, field_names, conditions=None):
    """
    Extract specified fields from a list of dictionaries with optional filtering.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        conditions: Optional dictionary specifying field conditions for filtering
    
    Returns:
        List of dictionaries containing only the specified fields that match conditions
    
    Raises:
        ValueError: If a condition references a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if condition fields exist in at least one record
    if conditions and data_list:
        all_fields = set()
        for record in data_list:
            all_fields.update(record.keys())
        
        for condition_field in conditions:
            if condition_field not in all_fields:
                raise ValueError(f"Condition references non-existent field: '{condition_field}'")
    
    # Filter records based on conditions
    filtered_data = []
    for record in data_list:
        match = True
        for field, value in conditions.items():
            if field not in record or record[field] != value:
                match = False
                break
        if match:
            filtered_data.append(record)
    
    # Extract only specified fields
    return [
        {field: record.get(field) for field in field_names}
        for record in filtered_data
    ]


def run_sql_query(records, command):
    """
    Execute a SQL-like query on a list of dictionaries.
    
    Args:
        records: List of dictionaries representing records
        command: SQL-like command string (supports SELECT, WHERE, ORDER BY)
    
    Returns:
        List of dictionaries containing query results
    
    Raises:
        ValueError: If the query is malformed or fails
    """
    if not command:
        raise ValueError("Query command cannot be empty")
    
    # Parse the SQL-like command
    command = command.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|$)', command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Query must start with SELECT")
    
    select_clause = select_match.group(1).strip()
    
    # Parse selected fields
    if select_clause == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    # Extract WHERE clause if present
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', command, re.IGNORECASE)
    where_conditions = []
    if where_match:
        where_clause = where_match.group(1).strip()
        # Parse conditions (simple format: field operator value)
        condition_pattern = r'(\w+)\s*(=|!=|>|<|>=|<=)\s*(.+?)(?:\s+AND\s+|$)'
        for match in re.finditer(condition_pattern, where_clause, re.IGNORECASE):
            field = match.group(1)
            op = match.group(2)
            value = match.group(3).strip()
            
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            else:
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            
            where_conditions.append((field, op, value))
    
    # Extract ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', command, re.IGNORECASE)
    order_field = None
    order_desc = False
    if order_match:
        order_field = order_match.group(1)
        if order_match.group(2) and order_match.group(2).upper() == 'DESC':
            order_desc = True
    
    # Apply WHERE filtering
    filtered_records = []
    for record in records:
        include = True
        for field, op, value in where_conditions:
            if field not in record:
                raise ValueError(f"Field '{field}' does not exist in records")
            
            record_value = record[field]
            
            # Apply operator
            if op == '=':
                if record_value != value:
                    include = False
                    break
            elif op == '!=':
                if record_value == value:
                    include = False
                    break
            elif op == '>':
                if not (record_value > value):
                    include = False
                    break
            elif op == '<':
                if not (record_value < value):
                    include = False
                    break
            elif op == '>=':
                if not (record_value >= value):
                    include = False
                    break
            elif op == '<=':
                if not (record_value <= value):
                    include = False
                    break
        
        if include:
            filtered_records.append(record)
    
    # Apply ORDER BY
    if order_field:
        if filtered_records and order_field not in filtered_records[0]:
            raise ValueError(f"Field '{order_field}' does not exist in records")
        filtered_records.sort(key=lambda x: x.get(order_field), reverse=order_desc)
    
    # Apply SELECT
    if selected_fields is None:
        result = filtered_records
    else:
        # Verify all selected fields exist
        if filtered_records:
            for field in selected_fields:
                if field not in filtered_records[0]:
                    raise ValueError(f"Field '{field}' does not exist in records")
        
        result = [
            {field: record.get(field) for field in selected_fields}
            for record in filtered_records
        ]
    
    return result
