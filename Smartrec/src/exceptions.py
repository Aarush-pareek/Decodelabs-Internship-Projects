"""Custom exceptions so the UI can show friendly messages."""


class SmartRecError(Exception):
    """Base class for all SmartRec errors."""


class DatasetError(SmartRecError):
    """The dataset could not be loaded or is unusable."""


class DatasetNotFoundError(DatasetError):
    """The dataset file does not exist."""


class DatasetValidationError(DatasetError):
    """The dataset exists but has a wrong structure or no valid rows."""


class InvalidPreferencesError(SmartRecError):
    """The user preferences are malformed or refer to unknown values."""


class EmptyPreferencesError(InvalidPreferencesError):
    """The user has not provided any preference signal."""


class SimilarityError(SmartRecError):
    """Similarity could not be computed (e.g. mismatched vector sizes)."""
