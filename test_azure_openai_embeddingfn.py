import os
import pytest
import inspect
from unittest.mock import patch
from openai import AzureOpenAI
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

def test_azure_openai_parameter_error():
    """
    Test to reproduce the bug where AzureOpenAI embedding function fails due to a parameter name mismatch.
    The AzureOpenAI client expects 'deployment' but ChromaDB passes 'azure_deployment'.
    """
    # Mock AzureOpenAI to intercept the parameters being used
    with patch('openai.AzureOpenAI', autospec=True) as mock_azure_client:
        # Setup the mock to capture arguments
        def record_args(*args, **kwargs):
            record_args.called_with = kwargs
            return mock_azure_client.return_value
        mock_azure_client.side_effect = record_args
        
        # Initialize the embedding function with Azure OpenAI configuration
        mock_api_key = "test_api_key"
        mock_api_version = "2023-05-15"
        mock_api_base = "https://test-openai-instance.openai.azure.com"
        mock_deployment_id = "test-embedding-deployment"
        
        OpenAIEmbeddingFunction(
            api_key=mock_api_key,
            api_type="azure",
            api_version=mock_api_version,
            api_base=mock_api_base,
            deployment_id=mock_deployment_id
        )
        
        # Check if AzureOpenAI was called with 'azure_deployment' instead of 'deployment'
        assert hasattr(record_args, 'called_with'), "AzureOpenAI constructor was not called"
        
        # Print out all parameters for debugging
        print(f"AzureOpenAI was called with: {record_args.called_with}")
        
        # Assert that 'azure_deployment' was used (which is wrong) and 'deployment' was not used
        assert 'azure_deployment' in record_args.called_with, "Expected incorrect parameter 'azure_deployment' not found"
        assert 'deployment' not in record_args.called_with, "Parameter 'deployment' incorrectly used"
        
        # Verify the deployment_id value is passed as expected
        assert record_args.called_with['azure_deployment'] == mock_deployment_id, "Incorrect deployment_id value"

# Bonus: Create a test that verifies the fix would work
def test_azure_openai_correct_parameter():
    """
    Test to verify the fix would work by manually replacing the parameter name.
    This test should fail until the fix is applied.
    """
    # Get the AzureOpenAI signature to check the correct parameter name
    signature = inspect.signature(AzureOpenAI.__init__)
    
    # Verify 'deployment' is a valid parameter and 'azure_deployment' is not
    print(f"AzureOpenAI constructor parameters: {list(signature.parameters.keys())}")
    assert 'deployment' in signature.parameters, "'deployment' should be a valid parameter for AzureOpenAI"
    assert 'azure_deployment' not in signature.parameters, "'azure_deployment' should not be a valid parameter for AzureOpenAI"
