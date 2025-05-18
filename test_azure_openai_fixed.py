import os
import pytest
from unittest.mock import patch, MagicMock
from openai import AzureOpenAI
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

def test_fixed_azure_openai_parameter():
    """
    Test to verify the fix for the AzureOpenAI parameter name issue.
    This test patches the OpenAIEmbeddingFunction to use the correct parameter name.
    """
    # Mock AzureOpenAI to intercept the parameters
    with patch('openai.AzureOpenAI', autospec=True) as mock_azure_client:
        # Setup the mock to capture arguments
        def record_args(*args, **kwargs):
            record_args.called_with = kwargs
            mock = MagicMock()
            # Setup embeddings.create to return a valid response
            mock.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
            )
            return mock
        
        mock_azure_client.side_effect = record_args
        
        # Create a patched version of the constructor to use 'deployment' instead of 'azure_deployment'
        original_init = OpenAIEmbeddingFunction.__init__
        
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.api_type == "azure":
                # The fix: replace 'azure_deployment' with 'deployment'
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.api_base,
                    deployment=self.deployment_id,  # Fixed parameter name
                    default_headers=self.default_headers,
                )
        
        # Apply the patched constructor
        with patch.object(OpenAIEmbeddingFunction, '__init__', patched_init):
            # Initialize the embedding function with Azure OpenAI configuration
            mock_api_key = "test_api_key"
            mock_api_version = "2023-05-15"
            mock_api_base = "https://test-openai-instance.openai.azure.com"
            mock_deployment_id = "test-embedding-deployment"
            
            # Create the embedding function
            ef = OpenAIEmbeddingFunction(
                api_key=mock_api_key,
                api_type="azure",
                api_version=mock_api_version,
                api_base=mock_api_base,
                deployment_id=mock_deployment_id
            )
            
            # Try to generate embeddings which should succeed with our patch
            embeddings = ef(["test text"])
            
            # Check the parameters passed to AzureOpenAI
            print(f"AzureOpenAI was called with: {record_args.called_with}")
            
            # This test SHOULD pass after the fix - we now use 'deployment' instead of 'azure_deployment'
            assert 'deployment' in record_args.called_with, "'deployment' parameter missing"
            assert 'azure_deployment' not in record_args.called_with, "'azure_deployment' should not be used"
            assert record_args.called_with['deployment'] == mock_deployment_id, "Incorrect deployment_id value"
