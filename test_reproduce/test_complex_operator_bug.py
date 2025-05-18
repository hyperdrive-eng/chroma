import pytest
import chromadb
import json

def test_complex_metadata_operator_bug():
    """
    This test demonstrates a critical bug in Chroma's metadata filtering when metadata keys match operator names.
    
    The issue occurs with complex where clauses that could be ambiguously interpreted.
    """
    client = chromadb.Client()
    collection = client.create_collection("test_collection_complex")
    
    # Setup test documents
    collection.add(
        ids=["id1", "id2", "id3", "id4"],
        documents=["doc1", "doc2", "doc3", "doc4"],
        metadatas=[
            {"$in": "value1", "category": "A"},           # Has a key that matches operator
            {"category": "A", "tag": "special"},          # Normal metadata, category A
            {"$in": "value3", "category": "B"},           # Has a key that matches operator, different category
            {"category": "B", "tag": "special"}           # Normal metadata, category B
        ]
    )
    
    print("=== Test: Complex filtering showing operator name ambiguity bug ===")
    
    # The critical test: filter for category B documents using "$in" as a METADATA KEY
    # and simultaneously filter for "tag" being in a list using $in as an OPERATOR
    try:
        bug_result = collection.get(
            where={"$and": [
                {"$in": "value3"},  # Looks for documents where metadata key "$in" equals "value3"
                {"category": "B"},   # Looks for documents where "category" is "B"
                {"tag": {"$in": ["special"]}}  # Uses $in as an operator to match "tag" in list
            ]}
        )
        
        print(f"Result of query trying to find documents with key '$in'='value3' AND category='B' AND tag in ['special']:")
        print(json.dumps(bug_result, indent=2))
        
        # Check if we get the expected result (should be only id4)
        # But due to the bug, the system may get confused with the "$in" usage
        if len(bug_result["ids"]) == 1 and bug_result["ids"][0] == "id4":
            print("✅ Query worked correctly, found the right document")
        else:
            print(f"❌ BUG CONFIRMED: Query returned incorrect results: {bug_result['ids']}")
            print("   Expected only id4, as it's the only one with category B AND tag 'special'")
            if len(bug_result["ids"]) == 0:
                print("   Got no results - system likely confused by '$in' appearing both as metadata key and operator")
    
    except Exception as e:
        print(f"❌ Error with complex query: {e}")
        print("BUG CONFIRMED: System cannot handle complex queries with operator names as metadata keys")
    
    # Compare with a reference query that doesn't use "$in" as a metadata key
    ref_result = collection.get(
        where={"$and": [
            {"category": "B"},
            {"tag": {"$in": ["special"]}}
        ]}
    )
    
    print("\nReference query without using '$in' as metadata key:")
    print(json.dumps(ref_result, indent=2))

if __name__ == "__main__":
    test_complex_metadata_operator_bug()
