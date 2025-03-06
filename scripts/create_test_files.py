#!/usr/bin/env python
"""
Script to create sample test files for testing file processors.
This script creates a test_files directory with sample files of different types.
"""
import os
import sys
from pathlib import Path
import shutil
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

# Test files directory
TEST_FILES_DIR = os.path.join(proj_root, "test_files")


def create_text_file(filename, content):
    """Create a text file with the given content."""
    filepath = os.path.join(TEST_FILES_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    logger.info(f"Created text file: {filepath}")


def create_test_files():
    """Create sample test files of different types."""
    # Create test_files directory if it doesn't exist
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    
    # Create a plain text file
    create_text_file("sample.txt", """
    This is a sample text file for testing the text processor.
    It contains multiple lines of text.
    The text processor should extract this text and provide basic metadata.
    """)
    
    # Create a markdown file
    create_text_file("sample.md", """
    # Sample Markdown File
    
    This is a **markdown** file with some _formatting_.
    
    ## Section 1
    
    - Item 1
    - Item 2
    - Item 3
    
    ## Section 2
    
    1. First item
    2. Second item
    3. Third item
    
    [Link to example](https://example.com)
    """)
    
    # Create a CSV file
    create_text_file("sample.csv", """
    Name,Age,Email
    John Doe,30,john@example.com
    Jane Smith,25,jane@example.com
    Bob Johnson,40,bob@example.com
    Alice Brown,35,alice@example.com
    """)
    
    logger.info(f"Created sample test files in {TEST_FILES_DIR}")
    logger.info("Note: For other file types (PDF, DOCX, XLSX, PPTX, images, audio, video), please add your own sample files to the test_files directory.")


if __name__ == "__main__":
    logger.info("Creating sample test files...")
    create_test_files()
    logger.info(f"Sample test files created in {TEST_FILES_DIR}")
    logger.info("For complete testing, please add your own PDF, DOCX, XLSX, PPTX, image, audio, and video files to the test_files directory.")