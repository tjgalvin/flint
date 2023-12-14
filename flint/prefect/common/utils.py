"""Common prefect related utilities that can be used between flows. 
"""
from typing import List, Any, TypeVar

from prefect import task

from flint.logging import logger

T = TypeVar('T')

@task 
def task_get_attributes(item: Any, attribute: str) -> Any:
    """Retrieve an attribute from an input instance of a class or structure. 
    
    This is intended to be used when dealing with a prefect future object that
    has yet to be evaluated or is otherwise not immediatedly accessible. 

    Args:
        item (Any): The item that has the input class or structure
        attribute (str): The attribute to extract

    Returns:
        Any: Vlue of the requested attribute
    """
    logger.debug(f"Pulling {attribute=}")
    return item.__dict__[attribute]

@task
def task_flatten(to_flatten: List[List[T]]) -> List[T]:
    """Will flatten a list of lists into a single list. This
    is useful for when a task-descorated function returns a list.


    Args:
        to_flatten (List[List[T]]): Input list of lists to flatten

    Returns:
        List[T]: Flattened form of input
    """
    logger.debug(f"Received {len(to_flatten)} to flatten.")
    
    flatten_list = [item for sublist in to_flatten for item in sublist]
    
    logger.debug(f"Flattened list {len(flatten_list)}")

    return flatten_list
