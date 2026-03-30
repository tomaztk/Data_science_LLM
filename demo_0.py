from openai import AzureOpenAI
from dotenv import load_dotenv
import os

"""
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    api_key=os.environ["AZURE_OAI_KEY"],
    api_version="2024-07-18",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o-mini")


def prompt_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Example usage
result = prompt_llm("What is the capital of France?")
print(result)

"""




from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os

load_dotenv()

# Use Azure AD token instead of API key
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,
    api_version="2024-02-01",
)
MODEL = os.environ.get("AZURE_OAI_DEPLOY", "gpt-4o-mini")

def prompt_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

result = prompt_llm("What is the capital of France?")
print(result)