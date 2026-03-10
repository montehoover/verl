import logging
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global task queue for multi-threaded processing
task_queue = Queue()


def create_task(task_identifier: str, task_content: dict) -> dict:
    """
    Creates a task dictionary from the given identifier and content.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        dict: A task dictionary with 'id' and 'content' keys.
    """
    # Create a structured task object with identifier and content
    return {
        'id': task_identifier,
        'content': task_content
    }


def enqueue_task(task: dict, queue: Queue) -> bool:
    """
    Adds a task to the specified queue.
    
    Args:
        task: The task dictionary to be enqueued.
        queue: The queue to add the task to.
    
    Returns:
        bool: True if the task is successfully enqueued, False otherwise.
    """
    try:
        # Attempt to add the task to the queue
        queue.put(task)
        
        # Log successful task enqueuing
        logger.info(
            f"Task successfully enqueued - ID: {task['id']}, "
            f"Content: {task['content']}"
        )
        
        return True
    except Exception as e:
        # Log any errors that occur during enqueuing
        logger.error(
            f"Failed to enqueue task - ID: {task['id']}, "
            f"Error: {str(e)}"
        )
        
        return False


def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.
    
    This is the main entry point for adding tasks to the system's task queue.
    It creates a properly formatted task and attempts to add it to the queue.
    
    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.
    
    Returns:
        bool: Returns True if the task is successfully enqueued, False if it fails to be added.
    """
    # Create a task object from the provided parameters
    task = create_task(task_identifier, task_content)
    
    # Attempt to enqueue the task to the global queue
    return enqueue_task(task, task_queue)
