import gradio as gr
import requests
import time
from requests.exceptions import HTTPError, RequestException

# URL of the API gateway (text-prompt endpoint)
url = "http://localhost:8001/api/text-prompt/"
retries = 3

class Chat:
    def __init__(self, limit=2):
        self.limit = limit
        self.conversation_history = []

    def get_prompt(self, message):
        self.conversation_history = self.conversation_history[-self.limit:]
        return "\n\n".join(self.conversation_history), f"User:\n{message}"

    def clear_history(self):
        """Clears the conversation history."""
        self.conversation_history = []

# Instantiate Chat once so that its state is preserved
C = Chat()

def test_text_processing(
    url,
    model_name,
    system_prompt,
    prompt,
    max_new_tokens,
    return_full_text,
    temperature,
    repeat_penalty,
    do_sample,
    think=False,
    top_k=0,
    min_p=0.0,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    seed=-1,
    clear_history=False,
    file=None,
):
    

    print("\n--- Debugging File Upload ---")
    print("Received file:", file)
    
    if clear_history:
        C.clear_history()
        print("Conversation history cleared.")
        
    history, payload = C.get_prompt(prompt)
    
    session = requests.Session()
    for attempt in range(1, retries + 1):
        
        
        try:
            
            
            if file is not None:
                with open(file.name, "rb") as f:
                    # files = {"file": f}
                    # response = session.post(
                    #     url, files=files, data=text_payload, timeout=250 * (attempt + 1)
                    # )
                    
                    input_prompt = f"{str(f.read())}\n\n{history}\n\n{str(payload)}" if history else f"{str(f.read())}\n\n\n{str(payload)}"
            else:
                input_prompt = f"{history}\n\n{str(payload)}" if history else str(payload)
                
            text_payload = {
                "model_name": model_name,
                "prompt": input_prompt,
                "system_prompt": system_prompt or "You are a highly knowledgeable, kind, and helpful assistant.",
                "temperature": temperature,
                "repeat_penalty": repeat_penalty,
                "think": think,
                "top_p": top_p,
            }

            # Only send optional params when set to non-default values
            if top_k > 0:
                text_payload["top_k"] = top_k
            if min_p > 0.0:
                text_payload["min_p"] = min_p
            if presence_penalty != 0.0:
                text_payload["presence_penalty"] = presence_penalty
            if frequency_penalty != 0.0:
                text_payload["frequency_penalty"] = frequency_penalty
            if seed >= 0:
                text_payload["seed"] = seed
            print(input_prompt[:1000])
            response = session.post(url, json=text_payload, timeout=250 * (attempt + 1))

            response.raise_for_status()  # Raise an HTTPError for bad responses
            json_resp = response.json()
            res = json_resp.get("response", "")
            print(json_resp)
            if not res:
                print(
                    "The API response did not contain content: %s", json_resp
                )
                raise ValueError("Invalid API response format.")

            C.conversation_history.append(payload)
            C.conversation_history.append(f"Agent:\n{res}\n\n\n")
            return json_resp
        except (HTTPError, RequestException) as err:
            print(f"Attempt {attempt}/{retries}: Error occurred: {err}")
            if attempt < retries:
                time.sleep(2**attempt)  # Pause before retrying
            else:
                return f"An error occurred after {retries} attempts: {err}"

    return "Unexpected error occurred."




def gradio_interface(
    model_name,
    system_message,
    prompt,
    max_new_tokens,
    return_full_text,
    temperature,
    repeat_penalty,
    do_sample,
    think,
    top_k,
    min_p,
    top_p,
    presence_penalty,
    frequency_penalty,
    seed,
    clear_history_checkbox,
    file,
):
    response = test_text_processing(
        url,
        model_name,
        system_message,
        prompt,
        max_new_tokens,
        return_full_text,
        temperature,
        repeat_penalty,
        do_sample,
        think,
        top_k,
        min_p,
        top_p,
        presence_penalty,
        frequency_penalty,
        seed,
        clear_history_checkbox,
        file,
    )
    if isinstance(response, dict):
        content = response.get("response", "")
        return content or "No content received from API"
    elif isinstance(response, str):
        return response
    else:
        return "An error occurred while processing the request."

# Model descriptions
model_descriptions = {
    "instruct": "Qwen3.5-35B-A3B (MoE): optimized for following instructions and general tasks.",
    "reasoning": "Reasoning model: optimized for complex reasoning and multi-step problems.",
}


css="""
    .gradio-container {
        background-color: #1e1e1e; 
        color: white;
        font-family: Arial, sans-serif;
    }
    .gr-button-primary {
        background-color: #4A90E2;
        color: white;
        border: none;
    }
    h1, p {
        color: #4A90E2;
    }
    .dynamic-textbox {
        resize: vertical;
        overflow: auto;
        max-height: 200px;
    }
"""

# Setting up the Gradio interface with custom styling and layout
with gr.Blocks(
    title="OctAgent💻",  # Update the page title here
    
) as interface:
    gr.Markdown(
        """
        <h1 style="text-align: center;">OctAgent 💻</h1>
        <p style="text-align: center;">An advanced general-purpose AI agent</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(choices=["instruct", "reasoning"], value="instruct", label="Model Name")
            model_info = gr.Textbox(
                value=model_descriptions["instruct"],
                label="Model Description",
                interactive=False,
            )
            max_new_tokens = gr.Slider(
                minimum=1, maximum=20480*4, step=1, value=2048, label="Max New Tokens"
            )
            return_full_text = gr.Checkbox(value=False, label="Return Full Text")
            temperature = gr.Slider(
                minimum=0, maximum=1, step=0.001, value=0.01, label="Temperature"
            )
            repeat_penalty = gr.Slider(
                minimum=1.0, maximum=2, step=0.01, value=1.2, label="Repeat Penalty"
            )
            do_sample = gr.Checkbox(value=True, label="Do Sample")
            think_checkbox = gr.Checkbox(value=False, label="Think (enable reasoning/thinking mode)")
            top_k = gr.Slider(
                minimum=0, maximum=200, step=1, value=0, label="Top-K (0 = disabled)"
            )
            min_p = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Min-P (0 = disabled)"
            )
            top_p = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Top-P"
            )
            presence_penalty = gr.Slider(
                minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Presence Penalty"
            )
            frequency_penalty = gr.Slider(
                minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Frequency Penalty"
            )
            seed_input = gr.Number(
                value=-1, label="Seed (-1 = random)", precision=0
            )
            clear_history_checkbox = gr.Checkbox(value=False, label="Clear History")
            file_input = gr.File(label="Upload a file (optional)")
            
        with gr.Column(scale=5):
            system_message = gr.Textbox(
                lines=30,
                label="System Message",
                placeholder="You are a knowledgeable assistant proficient in all areas of your field...",
                elem_classes="dynamic-textbox",
            )
            prompt = gr.Textbox(
                lines=50,
                label="User Prompt",
                placeholder="Ask a question or provide a prompt...",
                elem_classes="dynamic-textbox",
            )
            
            submit_btn = gr.Button("Generate Response", variant="primary")
            # examples = gr.Examples(
            #     examples=[
            #         [
            #             "You are an assistant who helps with cooking recipes.",
            #             "Can you provide a recipe for banana bread?",
            #             2048,
            #             False,
            #             0.7,
            #             True,
            #             None,
            #         ],
            #         [
            #             "You are a programming assistant.",
            #             "How do I write a Python function to reverse a string?",
            #             2048,
            #             False,
            #             0.5,
            #             True,
            #             None,
            #         ],
            #     ],
            #     inputs=[
            #         system_message,
            #         prompt,
            #         max_new_tokens,
            #         return_full_text,
            #         temperature,
            #         do_sample,
            #         clear_history_checkbox,
            #         file_input,
            #     ],
            # )
            output = gr.Markdown()

            submit_btn.click(
                gradio_interface,
                inputs=[
                    model_name,
                    system_message,
                    prompt,
                    max_new_tokens,
                    return_full_text,
                    temperature,
                    repeat_penalty,
                    do_sample,
                    think_checkbox,
                    top_k,
                    min_p,
                    top_p,
                    presence_penalty,
                    frequency_penalty,
                    seed_input,
                    clear_history_checkbox,
                    file_input,
                ],
                outputs=[output],
            )

            model_name.change(
                lambda x: model_descriptions[x],
                inputs=[model_name],
                outputs=[model_info],
            )

# Launch the interface
interface.launch(server_port=7862, css=css)