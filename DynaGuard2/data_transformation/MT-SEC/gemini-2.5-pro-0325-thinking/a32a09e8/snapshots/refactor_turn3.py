"""
Module for managing tasks and inserting them into a shared queue.

This module provides functionality to prepare tasks and add them to a global
task queue for parallel processing systems. It includes helper functions for
task preparation and queue insertion, aiding modularity and maintainability.
"""

from queue import Queue, Full


# task_queue: Global queue for task management and storage.
task_queue = Queue()


def _prepare_task_item(job_id: str, job_data: dict) -> tuple:
    """
    Prepares the task item from job_id and job_data.

    Args:
        job_id (str): The job's unique identifier.
        job_data (dict): The job's data.

    Returns:
        tuple: The prepared task item, typically (job_id, job_data).
    """
    return (job_id, job_data)


def _add_item_to_queue(queue_instance: Queue, task_item: tuple) -> bool:
    """
    Adds a prepared task item to the given queue instance.

    Args:
        queue_instance (Queue): The queue to add the item to.
        task_item (tuple): The task item to add.

    Returns:
        bool: True if the item was added successfully, False otherwise.
    """
    try:
        # Using put_nowait for a non-blocking call.
        # If queue_instance were initialized with a maxsize and became full,
        # this would raise queue.Full. For an unbounded Queue(), Full is not raised.
        queue_instance.put_nowait(task_item)
        return True
    except Full:
        # This handles the case where the queue might be full (if bounded).
        return False
    except Exception:
        # Catches any other unexpected error during the put operation.
        # In a production system, it would be good practice to log this error.
        return False


def insert_task_to_queue(job_id: str, job_data: dict) -> bool:
    """
    Facilitates task insertion into a shared queue within a parallel processing system.

    This function is tasked with incorporating new tasks into a global queue
    for subsequent execution. It uses helper functions to prepare the task
    and add it to the queue.

    Args:
        job_id (str): A unique string identifier assigned to each individual task.
        job_data (dict): A dictionary encompassing the task's details, parameters,
                         and other relevant information.

    Returns:
        bool: A boolean value indicating success (True) or failure (False) of
              task addition to the queue.
    """
    task_item = _prepare_task_item(job_id, job_data)
    return _add_item_to_queue(task_queue, task_item)
