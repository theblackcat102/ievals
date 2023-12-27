# Supported models

Closed source API : OpenAI, Azure, Anthropic, Gemini

Open weights models : we rely on tgi for inference, checkout **Text generation inference** for more details


## OpenAI

```bash
ieval gpt-3.5-turbo-0613 --api_key "<Your OpenAI platform Key>" --top_k 5
```

## Gemini Pro

```bash
ieval gemini-pro  --api_key "<Your API Key from https://ai.google.dev/>" --top_k 5
```

Currently we do not support models from vertex AI yet. So PaLM (bison) series are not supported

## Anthropic (instant, v1.3, v2.0)

```bash
ieval claude-instant-1  --api_key "<Anthropic API keys>"
```

## Azure OpenAI Model

```bash
export AZURE_OPENAI_ENDPOINT="https://XXXX.azure.com/"
ieval <your azure model name>  azure --api_key "<Your API Key>" --top_k 5
```

We haven't experimented with instruction based model from azure yet, so for instruction based models, you will have to fallback to openai's models

## DashScope

Before using models from dashscope please install it via pypi

```bash
pip install dashscope==1.13.6
```

Once installed, you should be able to run:

```bash
ieval <Your model name> --api_key "<Dash Scope API>"
```

Supported models : qwen-turbo, qwen-plus, qwen-max, qwen-plus-v1, bailian-v1

## Text generation inference

In order to reduce download friction, we recommend using [text-generation-inference](https://github.com/huggingface/text-generation-inference) for inferencing open-weight models

For example this would setup a simple tgi instance using docker

```bash
sudo docker run --gpus '"device=0"' \
    --shm-size 1g -p 8020:80 \
    -v /volume/saved_model/:/data ghcr.io/huggingface/text-generation-inference:1.1.0 \
    --max-input-length 4000 \
    --max-total-tokens 4096 \
    --model-id  GeneZC/MiniChat-3B
```
Note: For 5 shot settings, one might need to supply more than 5200 max-input-length to fit in the entire prompt

Once the server has warmed up, simply assign the models and IP:Port to the evaluation cli

```
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020
```

For custom models, you might need to provide tokens text for system, user, assistant and end of sentence.

```
ieval GeneZC/MiniChat-3B --ip_addr 0.0.0.0:8020 \
    --sys_token "<s> [|User|] " \
    --usr_token "<s> [|User|] " \
    --ast_token "[|Assistant|]" \
    --eos_token "</s>"
```

You can run `ieval supported` to check models which we have already included with chat prompt. (This feature will be deprecated once more models support format chat prompt function)
