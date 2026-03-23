<!-- docker compose up --no-deps api nginx --build -->
<!-- docker compose up --no-deps api nginx --build --scale api=3 -->

git clone --depth 1 https://github.com/keduka-ai/llm-api.git

/home/samusachi/WorkStation/keduka/aischool/llm-api

ssh psqthbxgr08wiz-6441173c@ssh.runpod.io -i ~/.ssh/kais_runpod 



scp -i ~/.ssh/kais_runpod -P 22165 -r /home/samusachi/WorkStation/keduka/aischool/llm-api root@194.68.245.204:/workspace
scp -i ~/.ssh/kais_runpod -P 22165 -r /home/samusachi/WorkStation/keduka/aischool/llm-api/runpod-deploy.sh root@194.68.245.204:/workspace/llm-api 
scp -i ~/.ssh/kais_runpod -P 22165 -r /home/samusachi/WorkStation/keduka/aischool/llm-api/.env root@194.68.245.204:/workspace/llm-api 


FORCE_BUILD=1 bash runpod-deploy.sh

https://194.68.245.204:8080/



# Instruct model — basic request
curl -X POST https://194.68.245.204:8001/api/chat/completions/ \
     -H 'Content-Type: application/json' \
     -d '{
           "model": "instruct",
           "messages": [
             {"role": "user", "content": "Write a Python function to sort a list."}
           ],
           "temperature": 0.7,
           "max_tokens": 256
         }'
# llm-api
Django + DRF API gateway that routes requests to llama-cpp-python backend servers. Supports multiple model types (instruct, reasoning) via an OpenAI-compatible interface, with a Gradio chat UI (`app.py`).
