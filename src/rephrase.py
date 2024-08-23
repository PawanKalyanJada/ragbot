from openai import OpenAI, AzureOpenAI
from config.prompts import REPHRASE_QUERY_PROMPT
from config.config import TEMPERATURE

class Rephrase:
    """
    A class to rephrase user queries based on provided context and history using OpenAI or Azure OpenAI.
    """

    def __init__(self, gpt_engine_name: str, api_key: str, azure_endpoint: str = None, api_version: str = None, openai_type: str = 'openai') -> None:
        """
        Initializes the Rephrase class with the necessary OpenAI or Azure OpenAI configurations.

        Args:
            gpt_engine_name (str): The name of the GPT model to be used.
            api_key (str): The API key for OpenAI or Azure OpenAI.
            azure_endpoint (str, optional): The Azure endpoint, required if using Azure OpenAI (default is None).
            api_version (str, optional): The API version for Azure OpenAI (default is None).
            openai_type (str): Specifies whether to use 'openai' or 'azure_openai' (default is 'openai').
        """
        self.gpt_engine_name = gpt_engine_name
        if openai_type == 'azure_openai':
            self.openai_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
        else:
            self.openai_client = OpenAI(api_key=api_key)

    def followup_query(self, query: str, history: str = None) -> str:
        """
        Generates a follow-up query based on the provided query and context history.

        Args:
            query (str): The user's natural language query.
            history (str, optional): The conversation history or context (default is None).

        Returns:
            str: The rephrased follow-up query.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model = self.gpt_engine_name,
                messages = [
                    {'role': 'system', 'content': REPHRASE_QUERY_PROMPT},
                    {'role': 'user', 'content': f'''Chat history:
--------
{history}
--------
User Query: ```{query}```
Rephrased question:'''}
                ],
                temperature = TEMPERATURE
            )
            rephrased_query = response.choices[0].message.content
            return rephrased_query
        except Exception as e:
            raise Exception(f"Error in follow-up: {e}")