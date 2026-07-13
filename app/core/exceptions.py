class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400, data: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.data = data
        super().__init__(message)


class BadRequestException(AppException):
    def __init__(self, message: str, data: dict | None = None):
        super().__init__(message=message, status_code=400, data=data)


class NotFoundException(AppException):
    def __init__(self, message: str, data: dict | None = None):
        super().__init__(message=message, status_code=404, data=data)


class InternalServerException(AppException):
    def __init__(self, message: str = "Lỗi hệ thống", data: dict | None = None):
        super().__init__(message=message, status_code=500, data=data)