"""
Test script for file processors.

This is not an automated test, but a manual test script to verify file processors work correctly.
It runs in an interactive mode, allowing you to specify individual tests or run all tests.
"""
import asyncio
import os
import sys
import json
from pathlib import Path
import importlib
import re
import time

# Add project root to Python path
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(proj_root))


async def test_processor(file_path: str, file_type: str, show_full_content: bool = False) -> None:
    """
    Test a file processor with a local file.
    
    Args:
        file_path: Path to the test file
        file_type: MIME type of the file
        show_full_content: Whether to show the full content or just a preview
    """
    try:
        start_time = time.time()
        # Import here to capture any updates to the module
        from app.services.file_processors import FileProcessorFactory
        
        print(f"\n--- Testing {file_type} processor with {file_path} ---")
        
        # Read the file
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Get the processor
        processor = FileProcessorFactory.get_processor(file_type)
        
        # Extract content
        print("Extracting content...")
        content = await processor.process(file_content, file_path)
        
        if show_full_content:
            print(f"Content:\n{content}")
        else:
            print(f"Content (first 500 chars): {content[:500]}...")
        
        # Extract metadata
        print("\nExtracting metadata...")
        metadata = await processor.get_metadata(file_content, file_path)
        print(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")
        
        end_time = time.time()
        print(f"\n--- Test completed for {file_type} ---\n")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error testing {file_type} processor: {e}")


def get_file_type_from_extension(ext: str) -> str:
    """
    Get the MIME type from a file extension.
    
    Args:
        ext: File extension (with dot)
        
    Returns:
        MIME type string or None if not supported
    """
    ext = ext.lower()
    
    mime_types = {
        ".txt": "text/plain",
        ".md": "text/markdown", 
        ".markdown": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".webm": "video/webm"
    }
    
    return mime_types.get(ext)


async def list_available_tests():
    """List all available test files in the test directory."""
    test_files_dir = os.path.join(proj_root, "test_files")
    
    # Ensure test directory exists
    os.makedirs(test_files_dir, exist_ok=True)
    
    # Check if any test files exist
    if not os.listdir(test_files_dir):
        print(f"No test files found in {test_files_dir}. Please add some files to test.")
        return []
    
    print("\nAvailable test files:")
    files = []
    for i, file_name in enumerate(sorted(os.listdir(test_files_dir))):
        file_path = os.path.join(test_files_dir, file_name)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        ext = os.path.splitext(file_name)[1].lower()
        mime_type = get_file_type_from_extension(ext)
        
        if mime_type:
            print(f"{i+1}. {file_name} ({mime_type})")
            files.append((file_name, file_path, mime_type))
        else:
            print(f"{i+1}. {file_name} (unsupported)")
            files.append((file_name, file_path, None))
    
    return files


async def run_all_tests():
    """Run all available tests."""
    # Import here to capture any updates to the module
    from app.services.file_processors import FileProcessorFactory
    
    test_files_dir = os.path.join(proj_root, "test_files")
    
    # Ensure test directory exists
    os.makedirs(test_files_dir, exist_ok=True)
    
    # Check if any test files exist
    if not os.listdir(test_files_dir):
        print(f"No test files found in {test_files_dir}. Please add some files to test.")
        return
    
    # Test processors with example files
    for file_name in sorted(os.listdir(test_files_dir)):
        file_path = os.path.join(test_files_dir, file_name)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Determine file type based on extension
        ext = os.path.splitext(file_name)[1].lower()
        mime_type = get_file_type_from_extension(ext)
        
        if mime_type:
            await test_processor(file_path, mime_type)
        else:
            print(f"Skipping file with unsupported extension: {file_name}")


async def interactive_loop():
    """Run an interactive loop for testing file processors."""
    while True:
        print("\n=== File Processor Testing Tool ===")
        print("Commands:")
        print("  list - List available test files")
        print("  all - Run all tests")
        print("  <number> - Run test for a specific file")
        print("  reload - Reload the FileProcessorFactory module")
        print("  exit/quit - Exit the testing tool")
        
        command = input("\nEnter command: ").strip().lower()
        
        if command in ["exit", "quit", "q"]:
            print("Exiting testing tool.")
            break
            
        elif command == "list":
            await list_available_tests()
            
        elif command == "all":
            print("Running all tests...")
            await run_all_tests()
            
        elif command == "reload":
            print("Reloading modules...")
            # Force reload of the module
            import app.services.file_processors
            importlib.reload(app.services.file_processors)
            print("Modules reloaded.")
            
        elif re.match(r'^\d+$', command):
            # Run a specific test
            files = await list_available_tests()
            try:
                index = int(command) - 1
                if 0 <= index < len(files):
                    file_name, file_path, mime_type = files[index]
                    if mime_type:
                        print(f"Running test for {file_name}...")
                        await test_processor(file_path, mime_type, show_full_content=True)
                    else:
                        print(f"File {file_name} has an unsupported extension.")
                else:
                    print(f"Invalid file number. Please enter a number between 1 and {len(files)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        else:
            print("Unknown command. Type 'list' to see available tests or 'exit' to quit.")


if __name__ == "__main__":
    print("Starting interactive file processor testing tool...")
    try:
        asyncio.run(interactive_loop())
    except KeyboardInterrupt:
        print("\nTesting tool interrupted. Exiting.")
    except Exception as e:
        print(f"Error in testing tool: {e}")