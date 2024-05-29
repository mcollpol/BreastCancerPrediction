import logging

from log_reg_model.config.core import PACKAGE_ROOT, config

# Adding only NullHandler to the library's loggers to avoid interfering with the application's logging configuration.
logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())


with open(PACKAGE_ROOT / "VERSION", encoding='utf-8') as version_file:
    __version__ = version_file.read().strip()
