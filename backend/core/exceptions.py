"""
Custom Exceptions
=================
Application-specific exceptions.
"""


class RemixFNDException(Exception):
    """Base exception for REMIX-FND."""
    pass


class ModelNotLoadedError(RemixFNDException):
    """Raised when trying to use a model that hasn't been loaded."""
    def __init__(self, model_name: str = "Model"):
        self.message = f"{model_name} not loaded. Call load_model() first."
        super().__init__(self.message)


class InvalidInputError(RemixFNDException):
    """Raised when input validation fails."""
    def __init__(self, message: str = "Invalid input"):
        super().__init__(message)


class FeatureDisabledError(RemixFNDException):
    """Raised when trying to use a disabled feature."""
    def __init__(self, feature_name: str):
        self.message = f"Feature '{feature_name}' is disabled"
        super().__init__(self.message)

