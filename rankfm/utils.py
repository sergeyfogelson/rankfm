"""
rankfm general utility functions
"""


class InputError(ValueError, AssertionError):
    """an argument or the model state is invalid for the requested operation

    it subclasses AssertionError as well as ValueError so that code written against earlier
    releases, which signalled these problems with [assert] statements, keeps catching it
    """


def require(condition, message):
    """raise InputError(message) unless [condition] holds; unlike [assert], this still runs under `python -O`"""

    if not condition:
        raise InputError(message)

def get_data(obj):
    """get the numeric data from either a pd.dataframe or np.ndarray

    :param obj: pd.dataframe or np.ndarray
    :return: the object's underlying np.ndarray data
    """

    if obj.__class__.__name__ in ('DataFrame', 'Series'):
        data = obj.values
    elif obj.__class__.__name__ == 'ndarray':
        data = obj
    else:
        raise TypeError("input data must be in either pd.dataframe/pd.series or np.ndarray format")
    return data
