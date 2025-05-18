#!/usr/bin/env python3
"""
ChromaDB Zero Cache Size Segmentation Fault Reproducer

This script reproduces the segmentation fault in ChromaDB that occurs
when hnsw_cache_size is zero due to limited file handles.
"""

import os
import tempfile
import shutil
import resource
from unittest.mock import patch
import chromadb


def reproduce_segfault():
    # Create a temporary directory for ChromaDB storage
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Mock a low file handle limit environment
        with patch('resource.getrlimit') as mock_getrlimit:
            # This will make hnsw_cache_size = 5 // 5 = 0
            mock_getrlimit.return_value = (5, 5)

            print("=== Reproducing zero cache size segmentation fault ===")
            print("Creating client with simulated low file handles...")
            client = chromadb.PersistentClient(path=temp_dir)

            print("Creating collection...")
            collection = client.create_collection("test_collection")

            print("Adding embedding data (this should trigger the segfault)...")
            # This will trigger the segmentation fault
            collection.add(
                embeddings=[[1.0, 2.0, 3.0]],
                documents=["test document"],
                ids=["id1"]
            )

            # If execution reaches here, the bug wasn't reproduced
            print("ERROR: Operation completed without segmentation fault!")
            print("Bug was not reproduced - expected a segmentation fault.")
            return False
    except Exception as e:
        # We either get a segfault (which terminates the program)
        # or possibly another exception related to the issue
        print(f"Exception: {str(e)}")
        return True
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("ChromaDB Zero Cache Size Segmentation Fault Reproducer")
    print("="*60)
    print("This script will attempt to reproduce the segmentation fault...")
    print("NOTE: Expected behavior is for the program to crash with a segmentation fault.")

    reproduce_segfault()
