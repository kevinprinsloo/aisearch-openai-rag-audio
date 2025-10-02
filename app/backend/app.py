import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    # Create app with increased payload size limit for audio files (10MB)
    app = web.Application(client_max_size=10*1024*1024)

    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
    rtmt.system_message = """
        You are a helpful assistant. Only answer questions based on information you searched in the knowledge base, accessible with the 'search' tool. 
        The user is listening to answers with audio, so it's *super* important that answers are as short as possible, a single sentence if at all possible. 
        Never read file names or source names or keys out loud. 
        Always use the following step-by-step instructions to respond: 
        1. Always use the 'search' tool to check the knowledge base before answering a question. 
        2. Always use the 'report_grounding' tool to report the source of information from the knowledge base. 
        3. Produce an answer that's as short as possible. If the answer isn't in the knowledge base, say you don't know.
    """.strip()

    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or None,
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.getenv("AZURE_SEARCH_USE_VECTOR_QUERY", "true") == "true")
        )

    rtmt.attach_to_app(app, "/realtime")
    
    # Import and add local voice endpoints
    from local_voice_backend import handle_process_audio, handle_health_check
    
    # Add local voice API routes
    app.router.add_post('/api/local-voice/process-audio', handle_process_audio)
    app.router.add_get('/api/local-voice/health', handle_health_check)

    current_directory = Path(__file__).parent
    
    # Serve static assets (CSS, JS, images, etc.)
    app.router.add_static('/assets', path=current_directory / 'static/assets', name='assets')
    
    # Serve specific static files from root (like favicon.ico, audio worklets, etc.)
    static_files = ['favicon.ico', 'audio-playback-worklet.js', 'audio-processor-worklet.js']
    for filename in static_files:
        file_path = current_directory / 'static' / filename
        if file_path.exists():
            async def serve_static(request, path=file_path):
                return web.FileResponse(path)
            app.router.add_get(f'/{filename}', serve_static)
    
    # Serve index.html for the root route
    async def serve_index(request):
        return web.FileResponse(current_directory / 'static/index.html')
    
    app.router.add_get('/', serve_index)
    
    # Catch-all route for client-side routing - serve index.html for React routes
    async def serve_react_app(request):
        # Serve index.html for all non-API, non-static file routes
        return web.FileResponse(current_directory / 'static/index.html')
    
    # Add specific React routes
    app.router.add_get('/local-voice-rag', serve_react_app)
    
    # Enable CORS for local voice endpoints
    @web.middleware
    async def cors_handler(request, handler):
        response = await handler(request)
        if request.path.startswith('/api/local-voice/'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_handler)
    
    # Handle OPTIONS requests for CORS
    async def options_handler(request):
        if request.path.startswith('/api/local-voice/'):
            return web.Response(
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        return web.Response(status=404)
    
    app.router.add_route('OPTIONS', '/api/local-voice/{path:.*}', options_handler)
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8767  # Changed from 8765 to avoid conflicts
    web.run_app(create_app(), host=host, port=port)
