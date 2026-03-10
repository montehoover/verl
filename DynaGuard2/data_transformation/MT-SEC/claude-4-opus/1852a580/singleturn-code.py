import re

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Generate a team roster for a specific department using a template.
    
    Args:
        division_name: The department name to filter by
        roster_template: The string template with placeholders like {employee.field}
        
    Returns:
        A formatted string with employee details for the specified department
        
    Raises:
        ValueError: If attempting to access restricted fields or on error
    """
    # Find all placeholders in the template
    placeholders = re.findall(r'\{employee\.(\w+)\}', roster_template)
    
    # Check for restricted fields
    restricted_fields = {'salary', 'personal_email'}
    for field in placeholders:
        if field in restricted_fields:
            raise ValueError(f"Access to field '{field}' is restricted")
    
    # Filter employees by department
    department_employees = [emp for emp in employees if emp.department == division_name]
    
    # Generate roster entries
    roster_entries = []
    for employee in department_employees:
        try:
            # Create a copy of the template for this employee
            entry = roster_template
            
            # Replace each placeholder
            for field in placeholders:
                if hasattr(employee, field):
                    value = getattr(employee, field)
                    entry = entry.replace(f'{{employee.{field}}}', str(value))
                else:
                    raise ValueError(f"Employee has no attribute '{field}'")
            
            roster_entries.append(entry)
            
        except Exception as e:
            raise ValueError(f"Error processing employee data: {str(e)}")
    
    # Join all entries with newlines
    return '\n'.join(roster_entries)
