class MeasureError(Exception):
    def __init__(self, location, message, **meta):
        self.location = location
        self.message = message
        self.meta = meta
    
    def _as_dict(self):
        return {'location': self.location, 'meta': self.meta, 'message': self.message, 'type': self.__class__.__name__}

class CleanerError(MeasureError):
    def __repr__(self):
        return f'Cleaning error at {self.location}: {self.message} ({self.meta})'

class MeasurementError(MeasureError):
    def __repr__(self):
        return f'Measuring error at {self.location}: {self.message} ({self.meta})'

class FileError(MeasureError):
    def __init__(self, message, **meta):
        self.message = message
        self.meta = meta
    
    def __repr__(self):
        return f'{self.message} ({self.meta})'
    
    def _as_dict(self):
        return {'meta': self.meta, 'message': self.message, 'type': self.__class__.__name__}

class SeparationError(MeasureError):
    def __repr__(self):
        return f'Separation error at {self.location}: {self.message} ({self.meta})'
