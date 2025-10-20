# utils/logger.py
import logging

class NoWatchdogFilter(logging.Filter):
    def filter(self, record):
        # Suppress specific noisy logs from watchdog's inotify_buffer
        if record.name.startswith('watchdog.observers.inotify_buffer'):
            return False
        return True

# Configure base logger (root) so all loggers inherit handlers.
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
file_handler.addFilter(NoWatchdogFilter())

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.addFilter(NoWatchdogFilter())

if not root_logger.hasHandlers():
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
else:
    # Avoid duplicate handlers during reloads.
    found_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    found_stream = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not found_file:
        root_logger.addHandler(file_handler)
    if not found_stream:
        root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)
