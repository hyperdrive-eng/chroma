#!/usr/bin/env python3
"""
ChromaDB Zero Cache Size Segmentation Fault Debugger

This script attempts to debug why the segmentation fault isn't being triggered
by examining how ChromaDB calculates and uses the hnsw_cache_size value.
"""

import os
import sys
import tempfile
import shutil
import resource
import platform
from unittest.mock import patch
import chromadb
import inspect


def print_method_source(obj, method_name):
    """Print the source code of a method for debugging"""
    if hasattr(obj, method_name):
        method = getattr(obj, method_name)
        print(f"=== Source code of {obj.__name__}.{method_name} ===")
        print(inspect.getsource(method))
        print("="*60)
    else:
        print(f"Method {method_name} not found in {obj.__name__}")


def debug_cache_size_calculation():
    """Debug how the cache size is calculated"""
    # Import the rust module to examine it
    import chromadb.api.rust as rust_module
    
    # Print info about the module
    print("=== ChromaDB Rust Module Information ===")
    print(f"Module location: {rust_module.__file__}")
    
    # Print the __init__ method to see how hnsw_cache_size is calculated
    print_method_source(rust_module, "__init__")
    
    # Get the current limits directly
    if platform.system() != "Windows":
        max_file_handles = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        print(f"Current max file handles: {max_file_handles}")
        print(f"Would calculate hnsw_cache_size as: {max_file_handles // 5}")
    
    # Check if we can see the Bindings class 
    if hasattr(rust_module, "Bindings"):
        print("=== Examining Bindings class ===")
        print_method_source(rust_module.Bindings, "__init__")
    else:
        print("Bindings class not found - it might be imported from elsewhere")


def trace_cache_size_processing():
    """Set up tracing to see how cache size is processed"""
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Import the module we need to patch
        import chromadb.api.rust as rust_module
        
        # Save original methods
        original_init = rust_module.__init__
        
        # Create patched version to trace hnsw_cache_size
        def patched_init(self, system):
            print(f"[TRACE] Entering patched rust_module.__init__")
            # Call original init
            original_init(self, system)
            
            # Print the calculated value
            if hasattr(self, 'hnsw_cache_size'):
                print(f"[TRACE] hnsw_cache_size calculated as: {self.hnsw_cache_size}")
            else:
                print("[TRACE] hnsw_cache_size attribute not found!")
        
        # Patch the Bindings.__init__ if available
        original_bindings_init = None
        if hasattr(rust_module, "Bindings"):
            original_bindings_init = rust_module.Bindings.__init__
            
            def patched_bindings_init(self, **kwargs):
                print(f"[TRACE] Entering patched Bindings.__init__")
                print(f"[TRACE] kwargs: {kwargs}")
                if 'hnsw_cache_size' in kwargs:
                    print(f"[TRACE] hnsw_cache_size from kwargs: {kwargs['hnsw_cache_size']}")
                    if kwargs['hnsw_cache_size'] == 0:
                        print("[TRACE] WARNING: Zero cache size detected!")
                
                # Call original method
                result = original_bindings_init(self, **kwargs)
                print("[TRACE] Bindings.__init__ completed")
                return result
            
            rust_module.Bindings.__init__ = patched_bindings_init
        
        # Apply the patch
        rust_module.__init__ = patched_init
        
        try:
            # Mock a low file handle limit
            with patch('resource.getrlimit') as mock_getrlimit:
                mock_getrlimit.return_value = (5, 5)
                
                print("[TEST] Creating client with mocked low file handles...")
                client = chromadb.PersistentClient(path=temp_dir)
                
                print("[TEST] Creating collection...")
                collection = client.create_collection("test_collection")
                
                print("[TEST] Attempting add operation (might segfault)...")
                try:
                    collection.add(
                        embeddings=[[1.0, 2.0, 3.0]],
                        documents=["test document"],
                        ids=["id1"]
                    )
                    print("[TEST] Add operation completed without segfault")
                    
                    # Try query as well
                    print("[TEST] Attempting query operation...")
                    collection.query(
                        query_embeddings=[[1.0, 2.0, 3.0]],
                        n_results=1
                    )
                    print("[TEST] Query operation completed without segfault")
                except Exception as e:
                    print(f"[TEST] Exception during operation: {str(e)}")
        finally:
            # Restore original methods
            rust_module.__init__ = original_init
            if original_bindings_init is not None:
                rust_module.Bindings.__init__ = original_bindings_init
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Debugger")
    print("="*60)
    
    # First debug how cache size is calculated
    debug_cache_size_calculation()
    
    # Then trace how it's processed
    trace_cache_size_processing()
