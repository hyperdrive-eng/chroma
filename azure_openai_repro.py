"""
Set API key when running test

```sh
$ AZURE_OPENAI_API_KEY="azure-openai-api-key" python <this_file>
```

Set `OPENAI_LOG=debug` to see more verbose debug logs

```sh
$ OPENAI_LOG=debug AZURE_OPENAI_API_KEY="azure-openai-api-key" python <this_file>
```
"""

import chromadb
import os
from openai import AzureOpenAI
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

"""
Constants
"""
# Source: https://portal.azure.com
AZURE_ENDPOINT = "https://chroma-repro-bug.openai.azure.com/"

# Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#overview
AZURE_MODEL_NAME = "text-embedding-3-small" 

# Source: https://portal.azure.com
DEPLOYMENT = "text-embedding-3-small"

# Source: https://portal.azure.com
API_VERSION = "2024-02-01"

# Source: https://github.com/openai/openai-python?tab=readme-ov-file#microsoft-azure-openai
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

# Creates a custom embedding function that uses our Azure OpenAI deployment
my_azure_openai_embedding_function = OpenAIEmbeddingFunction(
    api_type="azure",
    deployment_id=DEPLOYMENT,
    api_version=API_VERSION,
    api_base=AZURE_ENDPOINT,
    model_name=AZURE_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
)

# Creates a Chroma client
client = chromadb.Client()

# Creates a Chroma collection using our custom embedding function
collection = client.create_collection("my_collection", embedding_function=my_azure_openai_embedding_function)

# Embeds some example text
test = collection.add(
    ids=["cat_embedding", "apple_embedding", "san_francisco_embedding"],
    documents=["Cat", "Apple", "San Francisco"],
)

# Queries our collection for the nearest neighbors to the example text
nearest_neighbors = collection.query(
    query_texts=["Dog"],
    n_results=1,
)

# Prints the nearest neighbors
print(nearest_neighbors)