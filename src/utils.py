import logging
import sys

def setup_logging(name: str = "omni_rag"):
    """
    Sets up a structured logger that writes to console and app.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplication
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler("app.log", encoding='utf-8')
        file_handler.setLevel(logging.WARNING) # Log errors/warnings to file
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
