import re
from typing import Any
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def _search_tool(search_client: SearchClient, args: Any) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    # Keyword-based query using Azure AI Search
    search_results = await search_client.search(
        search_text=args['query'],
        query_type="simple",  # Using simple keyword search
        top=5,
        select="id,title,content"  # Updated to match the actual properties in your index
    )
    result = ""
    async for r in search_results:
        result += f"[{r['id']}]: {r['content']}\n-----\n"
    return ToolResult(result, ToolResultDirection.TO_SERVER)

# TODO: move from sending all chunks used for grounding eagerly to only sending links to 
# the original content in storage, it'll be more efficient overall
async def _report_grounding_tool(search_client: SearchClient, args: Any) -> None:
    sources = [s for s in args["sources"] if re.match(r'^[a-zA-Z0-9_-]+$', s)]
    list = " OR ".join(sources)
    print(f"Grounding source: {list}")
    # Use search instead of filter to align with how detailed integrated vectorization indexes
    search_results = await search_client.search(search_text=list, 
                                                select="id,title,content",  # Updated to match the actual properties in your index
                                                top=len(sources), 
                                                query_type="simple")  # Use simple for keyword-based search
    
    docs = []
    async for r in search_results:
        docs.append({"id": r['id'], "title": r["title"], "content": r['content']})
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier, search_endpoint: str, search_index: str, credentials: AzureKeyCredential | DefaultAzureCredential) -> None:
    if not isinstance(credentials, AzureKeyCredential):
        credentials.get_token("https://search.azure.com/.default") # warm this up before we start getting requests
    search_client = SearchClient(search_endpoint, search_index, credentials, user_agent="RTMiddleTier")

    rtmt.tools["search"] = Tool(schema=_search_tool_schema, target=lambda args: _search_tool(search_client, args))
    rtmt.tools["report_grounding"] = Tool(schema=_grounding_tool_schema, target=lambda args: _report_grounding_tool(search_client, args))
