from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)
import logging
import asyncio
import uuid
import json
import os

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential
from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import app_settings
from backend.utils import format_non_streaming_response, format_as_ndjson

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")
cosmos_db_ready = asyncio.Event()
logger = logging.getLogger('mylogger')
#set logging level
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('mylog.log')
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#write a debug message
logger.debug('This is a DEBUG message')
def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
    return app


# Dynamically define system prompts for IT and legal contexts
SYSTEM_PROMPTS = {
    "it": "You are an expert IT assistant. Provide detailed technical guidance and solutions for IT-related queries.",
    "legal": "You are a professional legal assistant. Provide accurate, concise, and compliant legal advice."
}


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/conversation/<string:context>", methods=["POST"])
async def conversation(context):
    """
    Handles conversation requests and dynamically selects the system prompt 
    based on the context path ('/it' or '/legal').
    """
    if not request.is_json:
        return jsonify({"error": "request must be JSON"}), 415

    request_json = await request.get_json()

    # Inject appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(context, app_settings.azure_openai.system_message)
    if not system_prompt:
        return jsonify({"error": f"Unsupported context '{context}'"}), 400

    # Add system prompt to the conversation's message history
    messages = request_json.get("messages", [])
    messages.insert(0, {"role": "system", "content": system_prompt})
    request_json["messages"] = messages

    # Process the conversation using the appropriate prompt
    return await conversation_internal(request_json, request.headers)


async def conversation_internal(request_body, request_headers):
    try:
        response, apim_request_id = await send_chat_request(request_body, request_headers)
        history_metadata = request_body.get("history_metadata", {})
        return jsonify(format_non_streaming_response(response, history_metadata, apim_request_id))
    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


async def send_chat_request(request_body, request_headers):
    """
    Sends a request to Azure OpenAI and returns the response.
    """
    filtered_messages = [
        message for message in request_body.get("messages", [])
        if message.get("role") != "tool"
    ]
    request_body["messages"] = filtered_messages

    model_args = prepare_model_args(request_body, request_headers)
    try:
        azure_openai_client = await init_openai_client()
        raw_response = await azure_openai_client.chat.completions.with_raw_response.create(**model_args)
        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id")
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id


def prepare_model_args(request_body, request_headers):
    """
    Prepares the model arguments for OpenAI's chat completion endpoint.
    """
    messages = request_body.get("messages", [])
    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model
    }
    return model_args


async def init_openai_client():
    """
    Initializes the AsyncAzureOpenAI client with Azure credentials.
    """
    try:
        endpoint = (
            app_settings.azure_openai.endpoint or
            f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )
        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=app_settings.azure_openai.key,
            azure_endpoint=endpoint,
        )
        return azure_openai_client
    except Exception as e:
        logging.exception("Failed to initialize Azure OpenAI client")
        raise e


async def init_cosmosdb_client():
    """
    Initializes the CosmosDB client if chat history is enabled.
    """
    if not app_settings.chat_history:
        logging.debug("Chat history is not enabled")
        return None

    try:
        cosmos_endpoint = f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
        credential = app_settings.chat_history.account_key or await DefaultAzureCredential()
        cosmos_conversation_client = CosmosConversationClient(
            cosmosdb_endpoint=cosmos_endpoint,
            credential=credential,
            database_name=app_settings.chat_history.database,
            container_name=app_settings.chat_history.conversations_container,
            enable_message_feedback=app_settings.chat_history.enable_feedback,
        )
        return cosmos_conversation_client
    except Exception as e:
        logging.exception("Failed to initialize CosmosDB client")
        raise e


app = create_app()
