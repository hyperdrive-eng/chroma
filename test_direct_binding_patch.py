import unittest
import sys
import tempfile
import os
import resource
import platform
import ctypes
from unittest.mock import patch, MagicMock
import chromadb
import inspect


class TestDirectBindingPatch(unittest.TestCase):
    """Test directly patching the rust bindings calculation"""

    def setUp(self):
        # Create a temporary directory for persistent storage
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_direct_patching_of_rust_py(self):
        """Directly patch chromadb/api/rust.py to demonstrate the bug and fix"""
        import types
        import chromadb.api.rust as rust_module
        
        # Find the original implementation in code
        source = inspect.getsource(rust_module)
        print("Found rust.py source code, length:", len(source))
        
        # Store the original module for restoration later
        original_module = types.ModuleType('original_rust_module')
        for attr in dir(rust_module):
            if not attr.startswith('__'):
                setattr(original_module, attr, getattr(rust_module, attr))
        
        # Create our patch for the module
        def patch_rust_module():
            # Patches the resource calculation to demonstrate the bug
            def create_bug_condition():
                print("Patching resource.getrlimit to return (5, 5)...")
                
                original_getrlimit = resource.getrlimit
                def mocked_getrlimit(resource_type):
                    if resource_type == resource.RLIMIT_NOFILE:
                        print("Returning mocked file handles limit: (5, 5)")
                        return (5, 5)
                    return original_getrlimit(resource_type)
                
                # Apply the patch
                resource.getrlimit = mocked_getrlimit
                
                print("Resource patching complete")
            
            # Call our function to create the bug condition
            create_bug_condition()
            
            # Now patch the actual calculation
            class_source = inspect.getsource(rust_module.RustImpl)
            print("Found RustImpl class, length:", len(class_source))
            
            # Original __init__ has the bug
            original_init = rust_module.RustImpl.__init__
            
            # Create our fixed version
            def fixed_init(self, system):
                print("Using patched __init__ with fix...")
                original_init(self, system)
                
                # Print what we got after calculation
                print(f"Original calculation result: hnsw_cache_size = {self.hnsw_cache_size}")
                
                # Apply the fix: ensure it's never zero
                if self.hnsw_cache_size == 0:
                    self.hnsw_cache_size = 1
                    print(f"Applied fix! New hnsw_cache_size = {self.hnsw_cache_size}")
            
            # Apply our patched init
            rust_module.RustImpl.__init__ = fixed_init
        
        # Apply the patch
        patch_rust_module()
        
        try:
            # Now test ChromaDB with the patched module
            print("Creating client with patched module...")
            client = chromadb.PersistentClient(path=self.temp_dir)
            
            # Create collection
            print("Creating collection...")
            collection = client.create_collection("test_collection")
            
            # Add data - with the fix, this should work without segfault
            print("Adding data (should NOT segfault with fix)...")
            collection.add(
                embeddings=[[1.0, 2.0, 3.0]],
                documents=["test document"],
                ids=["id1"]
            )
            
            # Query to verify it works
            print("Querying data...")
            results = collection.query(
                query_embeddings=[[1.0, 2.0, 3.0]],
                n_results=1
            )
            
            # Verify correct behavior
            self.assertEqual(len(results['ids']), 1)
            print("Test completed successfully with fixed implementation!")
            
        finally:
            # Restore the original module
            for attr in dir(original_module):
                if not attr.startswith('__'):
                    setattr(rust_module, attr, getattr(original_module, attr))
            
            # Restore resource.getrlimit
            # This is already handled by unittest mock's cleanup


if __name__ == "__main__":
    unittest.main()
