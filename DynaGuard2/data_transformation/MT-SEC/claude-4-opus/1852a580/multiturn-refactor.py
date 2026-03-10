import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def validate_template_fields(template: str) -> list:
    import re
    placeholders = re.findall(r'\{employee\.(\w+)\}', template)
    restricted_fields = ['salary', 'personal_email']
    
    for field in placeholders:
        if field in restricted_fields:
            logger.warning(f"Attempted access to restricted field: {field}")
            raise ValueError(f"Access to restricted field '{field}' is not allowed")
    
    logger.debug(f"Template fields validated: {placeholders}")
    return placeholders

def filter_employees_by_department(employees_list: list, department: str) -> list:
    filtered = [emp for emp in employees_list if emp.department == department]
    logger.debug(f"Filtered {len(filtered)} employees for department '{department}'")
    return filtered

def format_employee_roster(employees_list: list, template: str, fields: list) -> str:
    roster_lines = []
    
    for employee in employees_list:
        line = template
        for field in fields:
            try:
                value = getattr(employee, field)
                line = line.replace(f'{{employee.{field}}}', str(value))
            except AttributeError:
                logger.error(f"Unknown employee attribute: {field}")
                raise ValueError(f"Unknown employee attribute: {field}")
        roster_lines.append(line)
    
    logger.debug(f"Formatted roster for {len(roster_lines)} employees")
    return '\n'.join(roster_lines)

def generate_team_roster(division_name: str, roster_template: str) -> str:
    logger.info(f"Generating team roster for division: {division_name}")
    logger.debug(f"Template: {roster_template}")
    
    try:
        # Validate template and extract fields
        template_fields = validate_template_fields(roster_template)
        
        # Filter employees by department
        department_employees = filter_employees_by_department(employees, division_name)
        
        # Format and return the roster
        formatted_roster = format_employee_roster(department_employees, roster_template, template_fields)
        
        logger.info(f"Successfully generated roster for {len(department_employees)} employees in {division_name}")
        logger.debug(f"Final roster:\n{formatted_roster}")
        
        return formatted_roster
    
    except Exception as e:
        logger.error(f"Error generating roster for division '{division_name}': {str(e)}")
        raise
