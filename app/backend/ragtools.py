import re
from typing import Any
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
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
    # Hybrid + Reranking query using Azure AI Search
    search_results = await search_client.search(
        search_text=args['query'], 
        query_type="semantic",
        semantic_configuration_name="semantic_configuration",
        top=10,
        vector_queries=[VectorizableTextQuery(text=args['query'], k_nearest_neighbors=50, fields="text_vector")],
        select="chunk_id,title,chunk")
    result = ""
    async for r in search_results:
        result += f"[{r['chunk_id']}]: {r['chunk']}\n-----\n"
    return ToolResult(result, ToolResultDirection.TO_SERVER)

KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# TODO: move from sending all chunks used for grounding eagerly to only sending links to 
# the original content in storage, it'll be more efficient overall
async def _report_grounding_tool(search_client: SearchClient, args: Any) -> None:
    sources = [s for s in args["sources"] if KEY_PATTERN.match(s)]
    list = " OR ".join(sources)
    print(f"Grounding source: {list}")
    # Use search instead of filter to align with how detailt integrated vectorization indexes
    # are generated, where chunk_id is searchable with a keyword tokenizer, not filterable 
    search_results = await search_client.search(search_text=list, 
                                                search_fields=["chunk_id"], 
                                                select=["chunk_id", "title", "chunk"], 
                                                top=len(sources), 
                                                query_type="full",
                                                semantic_configuration_name="semantic_configuration")
    
    # If your index has a key field that's filterable but not searchable and with the keyword analyzer, you can 
    # use a filter instead (and you can remove the regex check above, just ensure you escape single quotes)
    # search_results = await search_client.search(filter=f"search.in(chunk_id, '{list}')", select=["chunk_id", "title", "chunk"])

    docs = []
    async for r in search_results:
        docs.append({"chunk_id": r['chunk_id'], "title": r["title"], "chunk": r['chunk']})
    return ToolResult({"sources": docs}, ToolResultDirection.TO_CLIENT)

def attach_rag_tools(rtmt: RTMiddleTier, search_endpoint: str, search_index: str, credentials: AzureKeyCredential | DefaultAzureCredential) -> None:
    if not isinstance(credentials, AzureKeyCredential):
        credentials.get_token("https://search.azure.com/.default") # warm this up before we start getting requests
    search_client = SearchClient(search_endpoint, search_index, credentials, user_agent="RTMiddleTier")

    rtmt.tools["search"] = Tool(schema=_search_tool_schema, target=lambda args: _search_tool(search_client, args))
    rtmt.tools["report_grounding"] = Tool(schema=_grounding_tool_schema, target=lambda args: _report_grounding_tool(search_client, args))
