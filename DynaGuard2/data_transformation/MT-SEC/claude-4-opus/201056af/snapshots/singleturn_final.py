import re

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build an employee directory for a specific department using a template.
    
    Args:
        team_name: The department name to filter by
        list_template: The string template with placeholders like {employee.field}
        
    Returns:
        A formatted string representing the employee directory
        
    Raises:
        ValueError: If restricted fields are accessed or an error occurs
    """
    # Check for restricted fields in the template
    restricted_fields = ['salary', 'personal_email']
    for field in restricted_fields:
        if f'employee.{field}' in list_template:
            raise ValueError(f"Access to field '{field}' is restricted")
    
    # Access the global employees list
    try:
        global employees
    except NameError:
        raise ValueError("Employees data not available")
    
    # Filter employees by department
    department_employees = [emp for emp in employees if emp.department == team_name]
    
    # Build the result
    result_lines = []
    
    for employee in department_employees:
        try:
            # Create a line from the template
            line = list_template
            
            # Find all placeholders in the format {employee.field}
            placeholders = re.findall(r'\{employee\.(\w+)\}', list_template)
            
            for field in placeholders:
                # Check if field is restricted
                if field in restricted_fields:
                    raise ValueError(f"Access to field '{field}' is restricted")
                
                # Check if field exists
                if not hasattr(employee, field):
                    raise ValueError(f"Field '{field}' does not exist")
                
                # Replace the placeholder with the actual value
                placeholder = f'{{employee.{field}}}'
                value = getattr(employee, field)
                line = line.replace(placeholder, str(value))
            
            result_lines.append(line)
            
        except Exception as e:
            # Re-raise as ValueError for any unexpected errors
            raise ValueError(f"Error processing template: {str(e)}")
    
    # Join all lines with newlines
    return '\n'.join(result_lines)
