#!/usr/bin/env python3
"""
ChromaDB Zero Cache Size Segmentation Fault Reproducer

This script attempts to reproduce the segfault by patching the Bindings
initialization to force a zero cache size at a lower level.
"""

import os
import tempfile
import shutil
import inspect
import platform
import sys
import chromadb
from unittest.mock import patch


def patch_bindings_initialization():
    """Patch the Bindings initialization to ensure zero cache size"""
    # First, find the actual Bindings class/module
    import importlib
    
    # Look through the chromadb package for rust bindings
    bindings_module = None
    for name in dir(chromadb):
        if "rust" in name.lower() or "bindings" in name.lower():
            try:
                mod = getattr(chromadb, name)
                if hasattr(mod, "Bindings"):
                    bindings_module = mod
                    print(f"Found Bindings in {name}")
                    break
            except (ImportError, AttributeError):
                pass
    
    if not bindings_module:
        # Try the known import
        try:
            import chromadb_rust_bindings
            bindings_module = chromadb_rust_bindings
            print("Found standalone chromadb_rust_bindings")
        except ImportError:
            pass
    
    if not bindings_module:
        print("Could not find Bindings module!")
        return None
    
    # Now patch the initialization
    original_bindings_init = bindings_module.Bindings.__init__
    
    def patched_bindings_init(self, *args, **kwargs):
        print("Patched Bindings.__init__ called")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        
        # Force hnsw_cache_size to 0
        if 'hnsw_cache_size' in kwargs:
            print(f"Original hnsw_cache_size: {kwargs['hnsw_cache_size']}")
            kwargs['hnsw_cache_size'] = 0
            print("Forced hnsw_cache_size to 0")
        
        # Call original
        return original_bindings_init(self, *args, **kwargs)
    
    bindings_module.Bindings.__init__ = patched_bindings_init
    
    return (bindings_module, original_bindings_init)


def patch_client_creation():
    """Alternative approach: intercept client creation"""
    import chromadb.api.rust as rust_module
    
    # Store original method
    if hasattr(rust_module, "_initialize_bindings"):
        original_method = rust_module._initialize_bindings
        
        def patched_initialize_bindings(self, *args, **kwargs):
            print("Patched _initialize_bindings called")
            # Force the cache size to 0
            if 'hnsw_cache_size' in kwargs:
                print(f"Original hnsw_cache_size: {kwargs['hnsw_cache_size']}")
                kwargs['hnsw_cache_size'] = 0
                print("Forced hnsw_cache_size to 0")
            
            return original_method(self, *args, **kwargs)
        
        rust_module._initialize_bindings = patched_initialize_bindings
        return (rust_module, original_method)
    
    return None


def inspect_rust_module():
    """Inspect the rust module to understand its structure"""
    import chromadb.api.rust as rust_module
    
    print("=== ChromaDB Rust Module Inspection ===")
    print(f"Module path: {rust_module.__file__}")
    print("Module contents:")
    for name in dir(rust_module):
        if not name.startswith("__"):
            item = getattr(rust_module, name)
            item_type = type(item).__name__
            print(f"  {name}: {item_type}")
            
            # If it's a class, show its methods
            if item_type == "type":
                print(f"    Methods:")
                for method_name in dir(item):
                    if not method_name.startswith("__"):
                        try:
                            method = getattr(item, method_name)
                            method_type = type(method).__name__
                            print(f"      {method_name}: {method_type}")
                        except Exception as e:
                            print(f"      {method_name}: Error - {str(e)}")


def reproduce_segfault():
    """Attempt to reproduce the segmentation fault"""
    # Create a temporary directory for ChromaDB storage
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Patch bindings
    patch_info = None
    client_patch_info = None
    
    try:
        # First inspect the module
        inspect_rust_module()
        
        # Try to patch at the bindings level
        patch_info = patch_bindings_initialization()
        
        # Also try to patch at client creation
        client_patch_info = patch_client_creation()
        
        print("=== Reproducing zero cache size segmentation fault ===")
        print("Creating client with patched bindings to force hnsw_cache_size=0...")
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
        print(f"Exception: {type(e).__name__} - {str(e)}")
        print("Got an exception rather than a segfault - likely related to the bug")
        
    finally:
        # Restore original methods
        if patch_info:
            patch_info[0].Bindings.__init__ = patch_info[1]
        
        if client_patch_info:
            client_patch_info[0]._initialize_bindings = client_patch_info[1]
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Reproducer (Bindings Patch)")
    print("="*80)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("This script will attempt to reproduce the segmentation fault by patching")
    print("the Bindings initialization to force hnsw_cache_size=0...")
    print("NOTE: Expected behavior is for the program to crash with a segmentation fault.")
    
    reproduce_segfault()
