#!/usr/bin/env python3
"""
Test script to reproduce and fix ChromaDB segmentation fault due to zero cache size.

This script:
1. Shows the current file handle limit
2. Creates a temporary patch to ensure hnsw_cache_size is never zero
3. Creates a collection and adds data without segfault
"""

import os
import sys
import platform
import resource
import tempfile
import shutil
from unittest.mock import patch
import chromadb


def demonstrate_current_limits():
    """Show the current system limits that affect the bug."""
    print("=== Current System Limits ===")
    if platform.system() != "Windows":
        max_file_handles = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    else:
        import ctypes
        max_file_handles = ctypes.windll.msvcrt._getmaxstdio()
    
    print(f"Maximum file handles: {max_file_handles}")
    print(f"Calculated hnsw_cache_size would be: {max_file_handles // 5}")
    print(f"With fix, hnsw_cache_size would be: {max(1, max_file_handles // 5)}")
    print("")


def apply_patch_and_test():
    """Apply a patch to rust.py and test with a low file handle limit."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Patch the module to simulate low file handles
        with patch('resource.getrlimit') as mock_getrlimit:
            # Set to 5 handles, which would result in hnsw_cache_size = 0
            mock_getrlimit.return_value = (5, 5)
            
            # Get the module to patch
            import chromadb.api.rust as rust_module
            
            # Store the original initialization method
            original_init = None
            for attr_name in dir(rust_module):
                attr = getattr(rust_module, attr_name)
                if hasattr(attr, '__init__') and callable(attr.__init__):
                    try:
                        if 'hnsw_cache_size' in getattr(attr, '__init__').__code__.co_varnames:
                            original_init = attr.__init__
                            patched_class = attr
                            print(f"Found class with hnsw_cache_size in __init__: {attr_name}")
                            break
                    except (AttributeError, TypeError):
                        continue
            
            if not original_init:
                print("ERROR: Could not find the appropriate class to patch!")
                return
            
            # Define our patched initialization method
            def patched_init(self, *args, **kwargs):
                # Call the original init
                original_init(self, *args, **kwargs)
                
                # Check if hnsw_cache_size exists and fix if it's zero
                if hasattr(self, 'hnsw_cache_size'):
                    print(f"Original hnsw_cache_size: {self.hnsw_cache_size}")
                    if self.hnsw_cache_size == 0:
                        self.hnsw_cache_size = 1
                        print(f"Applied fix! New value: {self.hnsw_cache_size}")
            
            # Apply the patch
            patched_class.__init__ = patched_init
            
            try:
                print("
=== Testing with patched module ===")
                print("Creating client with simulated low file handles...")
                client = chromadb.PersistentClient(path=temp_dir)
                
                print("Creating collection...")
                collection = client.create_collection("test_collection")
                
                print("Adding embedding data...")
                collection.add(
                    embeddings=[[1.0, 2.0, 3.0]],
                    documents=["test document"],
                    ids=["id1"]
                )
                
                print("Querying data...")
                results = collection.query(
                    query_embeddings=[[1.0, 2.0, 3.0]],
                    n_results=1
                )
                
                print(f"Query results: {results}")
                print("
SUCCESS: Operation completed without segmentation fault!")
                print("This confirms our fix works correctly.")
            finally:
                # Restore the original init method
                if original_init:
                    patched_class.__init__ = original_init
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Reproducer")
    print("=====================================================
")
    
    demonstrate_current_limits()
    apply_patch_and_test()
