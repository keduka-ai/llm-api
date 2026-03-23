import requests

# URL of the FastAPI server
url = "http://localhost:8000/api/text-prompt/"

def test_text_processing(
    url,
    system_prompt,
    prompt,
    max_new_tokens=8480,
    return_full_text=False,
    temperature=0.0005,
    model_name=4,
    do_sample=True,
):
    """
    Sends a POST request to the specified URL with the given system prompt and prompt.

    Args:
        url (str): The URL of the FastAPI server.
        system_prompt (str): The system prompt to be included in the request.
        prompt (str): The prompt to be processed.
        max_new_tokens (int): Maximum number of new tokens to generate.
        return_full_text (bool): Whether to return the full text.
        temperature (float): The sampling temperature.
        do_sample (bool): Whether to sample or not.

    Returns:
        dict: The response from the server if the request is successful.
        None: If the request fails.
    """
    # JSON payload
    text_payload = {
        "system_prompt": system_prompt,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "return_full_text": return_full_text,
        "temperature": temperature,
        "do_sample": do_sample,
        "model_name": f"{model_name}",
    }

    # Make the POST request
    response = requests.post(url, json=text_payload)

    # Handle the response
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


# Test Text Processing
print("\nTesting Text Processing...")

system_prompt = r"""You are an expert at extracting blocks of text or code from scripts, utilizing advanced techniques like regular expressions, abstract syntax trees (AST).Your proficiency spans various programming languages, enabling you to handle Python, JavaScript, C++, and more. You excel in both static and dynamic analysis, ensuring accurate extraction of relevant segments while preserving context and dependencies. Your meticulous approach guarantees high-quality results, making you a valuable resource for tasks requiring precise and efficient extraction of textual or code elements from scripts."""
prompt = r'''Given the following:

```
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch, os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
max_new_tokens = 20480

quantization_config = BitsAndBytesConfig(load_in_16bit=True)
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Ensure flash-attention is properly installed and used
try:
    import flash_attn
    logger.info("Flash-attention is installed and being used.")
except ImportError:
    logger.warning("Flash-attention is not installed or not being used correctly.")

# Initialize the model and processor paths
model_paths = {
    "128": {
        "tokenizer_path": "ai_api/tokenizers/phi3mini128-tokenizer",
        "model_path": "ai_api/models/phi3mini128-model"
    },
    # "4": {
    #     "tokenizer_path": "ai_api/tokenizers/phi3mini4-tokenizer",
    #     "model_path": "ai_api/models/phi3mini4-model"
    # },
    # "8": {
    #     "tokenizer_path": "ai_api/tokenizers/phi3mini4-tokenizer",
    #     "model_path": "ai_api/models/phi3mini4-model"
    # },
}

# Check to make sure models and tokenizers exist 
for key, paths in model_paths.items():
    tokenizer_path = paths['tokenizer_path']
    model_path = paths['model_path']   
    if not os.path.exists(tokenizer_path):
        raise NotADirectoryError(f"Tokenizer path does not exist: {tokenizer_path}")
    if not os.path.exists(model_path):
        raise NotADirectoryError(f"Model path does not exist: {model_path}")
    if not os.listdir(tokenizer_path):
        raise ValueError(f"Tokenizer path is empty: {tokenizer_path}")
    if not os.listdir(model_path):
        raise ValueError(f"Model path is empty: {model_path}")


def initialize_model_and_tokenizer(model_path, tokenizer_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize model on CPU first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,  # Use fp16 if GPU is available
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config
        )
        
        # Use DataParallel to utilize multiple GPUs
        # if torch.cuda.device_count() > 1:
        #     logger.info(f"Using {torch.cuda.device_count()} GPUs")
        #     model = torch.nn.DataParallel(model)

        # Move the model to GPU if available
        # model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {str(e)}")
        raise

# Load models and tokenizers into a dictionary
models = {}
for key, paths in model_paths.items():
    model, tokenizer, device = initialize_model_and_tokenizer(paths["model_path"], paths["tokenizer_path"])
    models[key] = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }

# Default system prompt message
default_system_prompt = {
    "role": "system", 
    "content": "You are a highly knowledgeable, kind, and helpful assistant."
}

class TextPromptView(APIView):

    def post(self, request, *args, **kwargs):
        try:
            # Seed the prompt with other similar prompts and results
            # seed = [
            #     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
            #     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
            # ]
            seed = []
            prompt = request.data.get('prompt', 'Hello!\n')
            system_prompt_content = request.data.get('system_prompt', default_system_prompt['content'])
            
            # Get generation arguments from request or use defaults
            generation_args = {
                "max_new_tokens": request.data.get('max_new_tokens', max_new_tokens),
                "return_full_text": request.data.get('return_full_text', False),
                "temperature": request.data.get('temperature', 0.0005),
                "do_sample": request.data.get('do_sample', True),
            }

            # Get the model name from the request or use default
            model_name = request.data.get('model_name', '128')
            if model_name not in models:
                model_name = '128'
            model_data = models[model_name]

            # Handle multi-GPU inference
            if isinstance(model_data["model"], torch.nn.DataParallel):
                model = model_data["model"]
            else:
                model = model_data["model"]

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=model_data["tokenizer"],
                # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use the correct device
            )

            # Create system prompt with the provided or default content
            # system_prompt = {"role": "system", "content": system_prompt_content}
            
            # Construct messages with the system prompt and user prompt
            messages = [*seed, {"role": "user", "content": f"{system_prompt_content}\n\n{prompt}"}]

            outputs = pipe(messages, **generation_args)
            response_text = outputs[0]['generated_text']

            return Response({"response": response_text}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in TextPromptView: {str(e)}", exc_info=True)
            return Response({"error": "An error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```
Extract content from above.
'title' is key, and value is content.
Make sure to return your response in the following format:
```
[{
"<title1>":"""
.
.
.
"""},
{
"<title2>":"""
.
.
.
"""},

]
```'''

response = test_text_processing(url, system_prompt, prompt)

# Print the response
print(response["response"])

print("\n")
