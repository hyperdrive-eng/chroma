#!/usr/bin/env python3
"""
ChromaDB Zero Cache Size Segmentation Fault Reproducer - Final Version

This script creates a minimal reproducible test case for the ChromaDB
zero-cache-size segmentation fault bug:

1. Forces hnsw_cache_size = 0 by directly patching the critical code path
2. Creates an empty collection and attempts operations that should trigger the bug
3. Properly checks if we're on Linux as the bug is Linux-specific
"""

import os
import tempfile
import shutil
import platform
import sys
import resource
from unittest.mock import patch
import chromadb


def create_direct_rust_patch():
    """
    Create a patch that directly sets hnsw_cache_size to zero
    in the ChromaDB Rust module.
    """
    # Import the rust module
    import chromadb.api.rust as rust_module
    
    # Define the patched class to replace RustBindingsAPI
    class ZeroCacheRustBindings(rust_module.RustBindingsAPI):
        """Subclass that forces hnsw_cache_size to zero"""
        
        def __init__(self, system):
            # Call parent constructor
            super().__init__(system)
            
            # Force cache size to zero to trigger the bug
            self.hnsw_cache_size = 0
            print(f"[PATCH] Forced hnsw_cache_size to zero")
    
    # Save original class
    original_class = rust_module.RustBindingsAPI
    
    # Replace with our patched version
    rust_module.RustBindingsAPI = ZeroCacheRustBindings
    
    return (rust_module, original_class)


def reproduce_segfault():
    """
    Attempt to trigger the segmentation fault with a direct patching approach
    """
    # Create a temporary directory for ChromaDB storage
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Apply our patch to force zero cache size
    import chromadb.api.rust as rust_module
    patch_info = create_direct_rust_patch()
    
    try:
        # Set ulimit artificially low to help trigger the issue
        if platform.system() == 'Linux':
            try:
                # Try to set a low ulimit
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                print(f"Current file limits: soft={soft}, hard={hard}")
                resource.setrlimit(resource.RLIMIT_NOFILE, (5, hard))
                print("Set soft limit to 5")
            except Exception as e:
                print(f"Could not set file limit: {e}")
        
        # Setup the client
        print("=== Creating client with zero cache size ===")
        client = chromadb.PersistentClient(path=temp_dir)
        
        print("Creating collection...")
        collection = client.create_collection("test_collection")
        
        print("Attempting add operation (should trigger segfault)...")
        try:
            # First operation on empty index should trigger the segfault
            collection.add(
                embeddings=[[1.0, 2.0, 3.0]],
                documents=["test document"],
                ids=["id1"]
            )
            print("Add operation succeeded - the bug wasn't triggered")
            
            # Try query as well since the bug might manifest in either operation
            print("Attempting query operation...")
            collection.query(
                query_embeddings=[[1.0, 2.0, 3.0]],
                n_results=1
            )
            print("Query operation succeeded - the bug wasn't triggered")
            
            print(f"Bug could not be reproduced on {platform.system()}")
            if platform.system() != 'Linux':
                print("NOTE: The segfault bug primarily affects Linux systems")
        except Exception as e:
            print(f"Exception during operation: {type(e).__name__} - {e}")
            print("Got an exception rather than a segfault")
            
    finally:
        # Restore the original class
        rust_module.RustBindingsAPI = patch_info[1]
        
        # Clean up the temp directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Reproducer")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("")
    print("This script attempts to reproduce the segmentation fault in ChromaDB")
    print("that occurs when hnsw_cache_size is set to zero on Linux systems.")
    print("Expected behavior on Linux: The program crashes with a segmentation fault")
    print("")
    
    # Run the reproducer
    reproduce_segfault()
