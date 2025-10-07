import sys


def error_message_detail(error: Exception, error_detail: sys):
    """
    This function returns a formatted error message string.

    Parameters:
        error (Exception): The exception object to be processed.
        error_detail (sys): The sys object containing the exception information.

    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception):
        """
        Constructor for the CustomException class.

        Parameters:
            error_message (Exception): The exception object to be processed.

        Returns:
            None
        """

        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=sys)

    def __str__(self):
        """
        Returns the error message as a string.

        Returns:
            str: The error message as a string.
        """
        return self.error_message
