from pydantic import ValidationError

__all__ = ["GemnineError", "ContentBlockedError", "ValidationError"]


class GemnineError(Exception):
    pass


class ContentBlockedError(GemnineError):
    def __init__(self, reason, *args: object) -> None:
        super().__init__(*args)
        self.reason = reason

    def __str__(self) -> str:
        return f"Content blocked because of {self.reason}"
