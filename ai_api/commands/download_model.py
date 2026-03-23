from pathlib import Path
import shutil
import time
import torch
from django.core.management.base import BaseCommand, CommandParser, CommandError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import TFAutoModelForCausalLM  # For TensorFlow models
from requests.exceptions import HTTPError
import requests
from huggingface_hub import snapshot_download

import config 

class Command(BaseCommand):
    """
    A Django management command to download and cache a Hugging Face model locally in a format suitable for use with llama.cpp.

    This command allows you to specify a Hugging Face model by name and download it to a specific
    directory on your local machine. If the directory already exists, it will be removed before the
    model is downloaded and saved as .bin.
    """
    
    help = "Download and cache a Hugging Face model locally in a .bin format."
    
    app_dir = Path(__file__).parent.parent.parent
    model_name = config.hub_model_name # 'microsoft/Phi-3.5-mini-instruct'
    save_model_path  = config.SAVE_MODELS_PATH # app_dir / "models/phi3mini-model"
    save_tokenizer_path = config.SAVE_TOKENIZER_PATH # app_dir / "tokenizers/phi3mini-tokenizer"
    
    
    def add_arguments(self, parser: CommandParser) -> None:
        """
        Adds command-line arguments for the management command.

        Args:
            parser: The argument parser provided by Django.
        """
        parser.add_argument(
            'model_name',
            type=str,
            nargs='?',
            default=self.model_name,
            help="The name of the Hugging Face model to download."
        )
        parser.add_argument(
            '--save_model_path',
            type=str,
            help="The directory path where the model should be saved.",
            default=self.save_model_path
        )
        parser.add_argument(
            '--save_tokenizer_path',
            type=str,
            help="The directory path where the tokenizer should be saved.",
            default=self.save_tokenizer_path
        )
        parser.add_argument(
            '--retry',
            type=int,
            help="Number of times to retry downloading the model and tokenizer if the first attempt fails.",
            default=1
        )
        
    def handle(self, *args, **options) -> str | None:
        
        """
        The main logic of the management command. This method is called when the command is executed.

        It handles downloading the specified Hugging Face model and saving it to the specified path.
        If the path exists, it removes the directory before downloading.

        Args:
            *args: Additional arguments.
            **options: Command-line options parsed by Django.
        """
        
        model_name = options.get('model_name')
        save_model_path = options.get('save_model_path')
        save_tokenizer_path = options.get('save_tokenizer_path')
        retries = options.get('retry')
        full_model = config.FULL_MODELS_PATH
        
        
       # Validate model_name
        if not model_name:
            raise CommandError("The 'model_name' argument is required and was not provided.")

        if not isinstance(model_name, str) or not model_name.strip():
            raise CommandError("The 'model_name' argument must be a non-empty string.")
        
        # Prepare save paths
        path_model = Path(save_model_path).resolve()
        path_tokenizer = Path(save_tokenizer_path).resolve()

        try:
            if path_model.exists() and path_model.is_dir():
                self.stdout.write(f"Directory '{path_model}' already exists. Removing it.")
                shutil.rmtree(path_model)
                shutil.rmtree(full_model)
            if path_tokenizer.exists() and path_tokenizer.is_dir():
                self.stdout.write(f"Directory '{path_tokenizer}' already exists. Removing it.")
                shutil.rmtree(path_tokenizer)

            path_model.mkdir(parents=True, exist_ok=True)
            path_tokenizer.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CommandError(f"Failed to prepare save directory '{path_model}': {e.strerror}")
            
        # Attempt to download model with retries
        attempt = 0
        while attempt <= retries:
            try:
                self.stdout.write(
                    self.style.NOTICE(
                        f"Attempt {attempt + 1} of downloading model '{model_name}' "
                        f"to path: '{path_model if path_model else 'default Hugging Face cache'}'"
                    )
                )

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",  # Ensure model is loaded on CPU
                    torch_dtype=torch.float16,  # Use FP16 to reduce file size, if supported
                    trust_remote_code=True, 
                )
                
                # Save model weights as .bin
                # torch.save(model.state_dict(), path_model / "pytorch_model.bin")

                # Save the config as well
                model.save_pretrained(save_model_path)
                snapshot_download(repo_id=model_name, local_dir=full_model, local_dir_use_symlinks=True)    

                tokenizer.save_pretrained(save_tokenizer_path)

                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully downloaded and saved the model '{model_name}' as .bin "
                        f"to '{save_model_path if save_model_path else 'default cache'}'"
                    )
                )
                break  # Exit loop if successful

            except (HTTPError, requests.exceptions.ConnectionError) as e:
                self.stderr.write(
                    self.style.ERROR(
                        f"Network error while downloading the model '{model_name}': {e}"
                    )
                )
                if attempt < retries:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Retrying download... ({attempt + 1}/{retries}) after 5 seconds."
                        )
                    )
                    time.sleep(5)  # Wait before retrying
                else:
                    raise CommandError(
                        f"Failed to download the model '{model_name}' after {retries + 1} attempts."
                    )
            except EnvironmentError as e:
                raise CommandError(
                    f"Error accessing the model '{model_name}'. "
                    f"Ensure the model name is correct and you have the necessary permissions. Details: {e}"
                )
            except ValueError as e:
                raise CommandError(
                    f"Invalid model name '{model_name}'. Please check the model name and try again. Details: {e}"
                )
            except Exception as e:
                raise CommandError(
                    f"An unexpected error occurred while downloading the model '{model_name}': {e}"
                )
            finally:
                attempt += 1
