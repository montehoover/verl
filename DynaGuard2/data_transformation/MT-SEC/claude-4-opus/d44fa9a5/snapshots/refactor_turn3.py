import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('employee_reports.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, worker, rating, feedback, bonus):
        self.worker = worker
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}

def get_employee(worker_id: str) -> Employee:
    """Retrieve employee data by worker ID."""
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    return employees[worker_id]

def get_performance(worker_id: str) -> Performance:
    """Retrieve performance data by worker ID."""
    return performances.get(worker_id)

def validate_template(template: str) -> None:
    """Validate that the template doesn't contain unauthorized fields."""
    if 'feedback' in template or 'bonus' in template:
        raise ValueError("Unauthorized access to feedback or bonus information")

def process_template(template: str, employee: Employee, performance: Performance = None) -> str:
    """Replace placeholders in template with actual data."""
    result = template
    
    # Replace employee placeholders
    result = result.replace('{worker.name}', employee.name)
    result = result.replace('{worker.position}', employee.position)
    result = result.replace('{worker.department}', employee.department)
    result = result.replace('{worker.emp_id}', employee.emp_id)
    
    # Replace performance placeholders if performance data exists
    if performance:
        result = result.replace('{assessment.rating}', str(performance.rating))
        result = result.replace('{assessment.worker.name}', performance.worker.name)
        result = result.replace('{assessment.worker.position}', performance.worker.position)
        result = result.replace('{assessment.worker.department}', performance.worker.department)
        result = result.replace('{assessment.worker.emp_id}', performance.worker.emp_id)
    
    return result

def create_employee_report(worker_id: str, template: str) -> str:
    """Create an employee report by replacing placeholders in the template."""
    logger.info(f"Starting report generation for worker ID: {worker_id}")
    
    try:
        validate_template(template)
        logger.debug(f"Template validation passed for worker ID: {worker_id}")
        
        employee = get_employee(worker_id)
        logger.debug(f"Employee data retrieved for worker ID: {worker_id}")
        
        performance = get_performance(worker_id)
        if performance:
            logger.debug(f"Performance data retrieved for worker ID: {worker_id}")
        else:
            logger.debug(f"No performance data found for worker ID: {worker_id}")
        
        report = process_template(template, employee, performance)
        logger.info(f"Report successfully generated for worker ID: {worker_id}")
        
        return report
        
    except ValueError as e:
        logger.error(f"Validation error for worker ID {worker_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report for worker ID {worker_id}: {str(e)}")
        raise
