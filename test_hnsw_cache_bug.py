import unittest
import sys
import tempfile
import os
import resource
import platform
import ctypes
from unittest.mock import patch, MagicMock
import chromadb
from chromadb.config import Settings
import inspect


class TestHNSWCacheBug(unittest.TestCase):
    """Test to reproduce the segmentation fault with zero-capacity cache"""

    def setUp(self):
        # Create a temporary directory for persistent storage
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_trace_cache_size_value(self):
        """Explicitly trace the hnsw_cache_size value to confirm it's being passed correctly"""
        import chromadb.api.rust as rust_module
        
        # Save original __init__
        original_init = rust_module.__init__
        original_bindings_init = None
        if hasattr(rust_module, "Bindings"):
            original_bindings_init = rust_module.Bindings.__init__
        
        # Patch __init__ to monitor the value
        def patched_init(self, system):
            print(f"Patching rust_module.__init__")
            # Call original init
            original_init(self, system)
            print(f"hnsw_cache_size calculated as: {self.hnsw_cache_size}")
        
        # Patch the bindings init if possible
        if hasattr(rust_module, "Bindings") and original_bindings_init:
            def patched_bindings_init(self, **kwargs):
                print(f"Bindings initialization with hnsw_cache_size: {kwargs.get('hnsw_cache_size')}")
                if kwargs.get('hnsw_cache_size') == 0:
                    print("WARNING: Zero cache size detected! This would cause segfault.")
                return original_bindings_init(self, **kwargs)
            
            rust_module.Bindings.__init__ = patched_bindings_init
        
        # Apply the patch
        rust_module.__init__ = patched_init
        
        try:
            # Now force a very low file handle count
            with patch('resource.getrlimit') as mock_getrlimit:
                # Limit to 5 file handles, so hnsw_cache_size = 5 // 5 = 0
                mock_getrlimit.return_value = (5, 5)
                
                # Create client
                print("Creating client with mocked low file handles...")
                client = chromadb.PersistentClient(path=self.temp_dir)
                
                print("Creating collection...")
                collection = client.create_collection("test_collection")
                
                print("Adding data...")
                try:
                    collection.add(
                        embeddings=[[1.0, 2.0, 3.0]],
                        documents=["test document"],
                        ids=["id1"]
                    )
                    print("Successfully added data without segfault, but we expected one!")
                except Exception as e:
                    print(f"Exception during add: {str(e)}")
                    # We expect either a segfault (crash) or exception related to zero cache
                    print("Test successfully triggered the issue")
                    raise
        finally:
            # Restore original methods
            rust_module.__init__ = original_init
            if hasattr(rust_module, "Bindings") and original_bindings_init:
                rust_module.Bindings.__init__ = original_bindings_init
    
    def test_fix_implementation(self):
        """Implement and test the fix directly."""
        import chromadb.api.rust as rust_module
        
        # Save original __init__
        original_init = rust_module.__init__
        
        # Create patched version with fix
        def patched_init_with_fix(self, system):
            # Call original init
            original_init(self, system)
            
            # Apply the fix - ensure hnsw_cache_size is at least 1
            print(f"Original hnsw_cache_size: {self.hnsw_cache_size}")
            self.hnsw_cache_size = max(1, self.hnsw_cache_size)
            print(f"Fixed hnsw_cache_size: {self.hnsw_cache_size}")
        
        # Apply the patch
        rust_module.__init__ = patched_init_with_fix
        
        try:
            # Force a very low file handle count
            with patch('resource.getrlimit') as mock_getrlimit:
                # Limit to 5 file handles, which would create a 0 cache size without the fix
                mock_getrlimit.return_value = (5, 5)
                
                # Create client
                print("Creating client with mocked low file handles + fix...")
                client = chromadb.PersistentClient(path=self.temp_dir)
                
                print("Creating collection with fix applied...")
                collection = client.create_collection("test_collection_fixed")
                
                print("Adding data with fix applied...")
                collection.add(
                    embeddings=[[1.0, 2.0, 3.0]],
                    documents=["test document"],
                    ids=["id1"]
                )
                
                print("Querying data with fix applied...")
                results = collection.query(
                    query_embeddings=[[1.0, 2.0, 3.0]],
                    n_results=1
                )
                
                self.assertEqual(len(results['ids']), 1)
                print("Test with fix applied successfully completed!")
        finally:
            # Restore original method
            rust_module.__init__ = original_init


if __name__ == "__main__":
    unittest.main()
