#!/usr/bin/env python3
"""
ChromaDB Zero Cache Size Segmentation Fault Reproducer

This script reproduces the segmentation fault in ChromaDB that occurs
when hnsw_cache_size is zero by directly monkey patching the relevant class.
"""

import os
import tempfile
import shutil
import types
import chromadb


def monkey_patch_for_zero_cache():
    """
    Directly patch the ChromaDB Rust bindings to force hnsw_cache_size to zero
    """
    # Import the rust module
    import chromadb.api.rust as rust_module
    
    # Store the original __init__ method
    original_init = rust_module.__init__
    
    # Define our patched version that forces hnsw_cache_size to 0
    def patched_init(self, system):
        # Call the original __init__
        original_init(self, system)
        
        # Force hnsw_cache_size to 0 to trigger the bug
        print(f"Original hnsw_cache_size: {self.hnsw_cache_size}")
        self.hnsw_cache_size = 0
        print(f"Forced hnsw_cache_size to 0 to trigger the bug")
    
    # Apply our patch
    rust_module.__init__ = types.MethodType(patched_init, rust_module)
    
    return original_init


def reproduce_segfault():
    """Attempt to reproduce the segmentation fault"""
    # Create a temporary directory for ChromaDB storage
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Apply our monkey patch
    import chromadb.api.rust as rust_module
    original_init = monkey_patch_for_zero_cache()
    
    try:
        print("=== Reproducing zero cache size segmentation fault ===")
        print("Creating client with directly patched hnsw_cache_size=0...")
        client = chromadb.PersistentClient(path=temp_dir)
        
        print("Creating collection...")
        collection = client.create_collection("test_collection")
        
        print("Adding embedding data (this should trigger the segfault)...")
        # This will trigger the segmentation fault if the bug exists
        collection.add(
            embeddings=[[1.0, 2.0, 3.0]],
            documents=["test document"],
            ids=["id1"]
        )
        
        # If execution reaches here, the bug wasn't reproduced
        print("ERROR: Operation completed without segmentation fault!")
        print("Bug was not reproduced - expected a segmentation fault.")
        
        # Try query as well
        print("Trying query operation...")
        collection.query(
            query_embeddings=[[1.0, 2.0, 3.0]],
            n_results=1
        )
        print("Query completed without segmentation fault!")
        
    except Exception as e:
        # We might get an exception instead of a segfault
        print(f"Exception: {str(e)}")
        print("Got an exception rather than a segfault - likely related to the bug")
        
    finally:
        # Restore the original __init__
        rust_module.__init__ = original_init
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Reproducer (Direct Patching)")
    print("="*80)
    print("This script will attempt to reproduce the segmentation fault by directly")
    print("modifying the hnsw_cache_size to zero...")
    print("NOTE: Expected behavior is for the program to crash with a segmentation fault.")
    
    reproduce_segfault()
