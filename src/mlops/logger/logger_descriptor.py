from src.mlops.logger.utils.logger import Log

class LoggerDescriptor:

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, "_logger"):
            # Use the name of the subclass as the logger name
            logger_name = owner.__name__
            instance._logger = Log(logger_name).getlog()
        return instance._logger
