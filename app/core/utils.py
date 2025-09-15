"""
Path and System Utilities

Provides consistent path handling and system utilities across the application.
Python equivalent of the Node.js utils.js module.
"""

import os
import sys
import inspect
from pathlib import Path
from typing import Union, List, Optional


def get_dirname(file_path: Optional[str] = None) -> str:
    """
    Get directory name of the calling module or specified file path
    
    Args:
        file_path: Optional file path. If None, uses caller's file path
        
    Returns:
        Directory path of the module
    """
    if file_path is None:
        # Get the caller's file path
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
    
    return os.path.dirname(os.path.abspath(file_path))


def get_filename(file_path: Optional[str] = None) -> str:
    """
    Get absolute file path of the calling module or specified file path
    
    Args:
        file_path: Optional file path. If None, uses caller's file path
        
    Returns:
        Absolute file path of the module
    """
    if file_path is None:
        # Get the caller's file path
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
    
    return os.path.abspath(file_path)


def safe_path(*paths: Union[str, Path]) -> str:
    """
    Safely join paths, ensuring all arguments are strings
    
    Args:
        *paths: Path segments to join
        
    Returns:
        Joined path as string
    """
    # Convert all paths to strings
    string_paths = []
    for p in paths:
        if isinstance(p, Path):
            string_paths.append(str(p))
        else:
            string_paths.append(str(p))
    
    return os.path.join(*string_paths)


def safe_resolve(*paths: Union[str, Path]) -> str:
    """
    Safely resolve paths to absolute path, ensuring all arguments are strings
    
    Args:
        *paths: Path segments to resolve
        
    Returns:
        Resolved absolute path as string
    """
    # Convert all paths to strings
    string_paths = []
    for p in paths:
        if isinstance(p, Path):
            string_paths.append(str(p))
        else:
            string_paths.append(str(p))
    
    return os.path.abspath(os.path.join(*string_paths))


def is_main_module(file_path: Optional[str] = None) -> bool:
    """
    Check if a module is being run directly (equivalent to __name__ == '__main__')
    
    Args:
        file_path: Optional file path. If None, uses caller's file path
        
    Returns:
        True if the module is being run directly
    """
    if file_path is None:
        # Get the caller's file path
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
    
    # Get the main script path
    main_script = os.path.abspath(sys.argv[0]) if sys.argv else None
    current_file = os.path.abspath(file_path)
    
    return main_script == current_file


def get_project_root(file_path: Optional[str] = None) -> str:
    """
    Get project root directory by searching for common project files
    
    Args:
        file_path: Optional file path to start search from. If None, uses caller's file path
        
    Returns:
        Project root directory path
    """
    if file_path is None:
        # Get the caller's file path
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
    
    current_dir = os.path.dirname(os.path.abspath(file_path))
    
    # Common project root indicators
    root_indicators = [
        'package.json',
        'requirements.txt',
        'pyproject.toml',
        'setup.py',
        '.git',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    # Navigate up to find project root
    dir_path = current_dir
    while dir_path != os.path.dirname(dir_path):  # Not at filesystem root
        for indicator in root_indicators:
            indicator_path = os.path.join(dir_path, indicator)
            if os.path.exists(indicator_path):
                return dir_path
        
        dir_path = os.path.dirname(dir_path)
    
    # Fallback to current directory
    return current_dir


def ensure_directory(dir_path: Union[str, Path]) -> str:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        dir_path: Directory path to ensure exists
        
    Returns:
        Absolute path of the directory
    """
    abs_path = os.path.abspath(str(dir_path))
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_relative_path(file_path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> str:
    """
    Get relative path from base path to file path
    
    Args:
        file_path: Target file path
        base_path: Base path to calculate relative from. If None, uses current working directory
        
    Returns:
        Relative path as string
    """
    if base_path is None:
        base_path = os.getcwd()
    
    return os.path.relpath(str(file_path), str(base_path))


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize a path by resolving . and .. components and converting to absolute path
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized absolute path
    """
    return os.path.normpath(os.path.abspath(str(path)))


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension from file path
    
    Args:
        file_path: File path to extract extension from
        
    Returns:
        File extension (including the dot), or empty string if no extension
    """
    return os.path.splitext(str(file_path))[1]


def get_file_basename(file_path: Union[str, Path], include_extension: bool = True) -> str:
    """
    Get base name of file from file path
    
    Args:
        file_path: File path to extract basename from
        include_extension: Whether to include file extension
        
    Returns:
        File basename
    """
    basename = os.path.basename(str(file_path))
    if not include_extension:
        basename = os.path.splitext(basename)[0]
    return basename


def list_files(directory: Union[str, Path], pattern: Optional[str] = None, recursive: bool = False) -> List[str]:
    """
    List files in a directory, optionally with pattern matching
    
    Args:
        directory: Directory to search in
        pattern: Optional glob pattern to match files
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    import glob
    
    dir_path = str(directory)
    
    if pattern:
        if recursive:
            search_pattern = os.path.join(dir_path, '**', pattern)
            return glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(dir_path, pattern)
            return glob.glob(search_pattern)
    else:
        files = []
        if recursive:
            for root, dirs, filenames in os.walk(dir_path):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        files.append(item_path)
            except (OSError, FileNotFoundError):
                pass
        
        return files


# Default export equivalent
__all__ = [
    'get_dirname',
    'get_filename', 
    'safe_path',
    'safe_resolve',
    'is_main_module',
    'get_project_root',
    'ensure_directory',
    'get_relative_path',
    'normalize_path',
    'get_file_extension',
    'get_file_basename',
    'list_files'
]
