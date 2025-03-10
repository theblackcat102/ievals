# iEvals : iKala's Evaluator for Large Language Models

<p align="center"> <img src="resources/ieval_cover.png" style="width: 50%; max-width: 400px" id="title-icon">       </p>

Note: to all users, who wish to benchmark their own models, please use VLLM or sglang or text-generation-inference to setup an inference endpoint and export the variable CUSTOM_API_URL="your-endpoint-link/v1" and use '--series openai_chat' when evaluating

For example using openrouter openai compatible url

```
export CUSTOM_API_URL="https://openrouter.ai/api/v1"
ieval --series openai_chat --api_key XXX <your model name>
```

This is much faster and the only official method to benchmark later models


iEvals is a framework for evaluating chinese large language models (LLMs), especially performance in traditional chinese domain. Our goal was to provide an easy to setup and fast evaluation library for guiding the performance/use on existing chinese LLMs.

Currently, we only support evaluation for [TMMLU+](https://huggingface.co/datasets/ikala/tmmluplus), however in the future we are exploring more domain, ie knowledge extensive dataset (CMMLU, C-Eval) as well as context retrieval and multi-conversation dataset.


# Updated Leaderboard

                   Model                    | humanities | social sciences |   STEM   |  Others  | Average 
-----------------------------------------------------------------------------------------------------------
deepseek-chat                               |  73.19   |  81.93   |  82.93   |  74.41   |  78.11  
Qwen/Qwen2.5-72B-Instruct-Turbo             |  67.59   |  79.36   |  82.57   |  72.65   |  75.54  
gpt-4o-2024-08-06                           |  65.48   |  78.23   |  81.39   |  71.24   |  74.08  
claude-3-5-sonnet-20240620                  |  73.23   |  78.27   |  68.50   |  69.35   |  72.34  
gemini-2.0-flash-lite-preview-02-05         |  64.66   |  73.48   |  70.00   |  63.90   |  68.01  
Qwen/QwQ-32B-Preview                        |  57.98   |  70.94   |  72.87   |  63.59   |  66.35  
claude-3-opus-20240229                      |  60.34   |  70.12   |  67.43   |  62.32   |  65.05  
gemini-1.5-pro                              |  61.84   |  70.29   |  66.18   |  60.30   |  64.65  
gpt-4o-mini-2024-07-18                      |  55.01   |  67.09   |  73.16   |  61.36   |  64.15  
mistralai/Mistral-Small-24B-Instruct-2501   |  54.56   |  68.32   |  73.25   |  59.25   |  63.85  
llama-3.1-70b-versatile                     |  64.94   |  70.14   |  58.63   |  61.33   |  63.76  
Qwen/Qwen2.5-7B-Instruct-Turbo              |  54.42   |  64.51   |  68.01   |  58.83   |  61.44  
yentinglin/Llama-3-Taiwan-8B-Instruct       |  61.51   |  67.61   |  52.05   |  58.60   |  59.94  
claude-3-sonnet-20240229                    |  52.06   |  59.38   |  49.87   |  51.64   |  53.24  
Qwen2-7B-Instruct                           |  55.66   |  66.40   |  27.18   |  55.32   |  51.14  
gemma2-9b-it                                |  45.38   |  55.76   |  49.89   |  48.92   |  49.99  
claude-3-haiku-20240307                     |  47.48   |  54.48   |  48.47   |  48.77   |  49.80  
gemini-1.5-flash                            |  42.99   |  53.42   |  53.47   |  46.56   |  49.11  
reka-flash                                  |  44.07   |  52.68   |  46.04   |  43.43   |  46.56  
meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo |  44.03   |  50.95   |  42.75   |  45.19   |  45.73  
mixtral-8x7b-32768                          |  44.75   |  50.34   |  32.60   |  43.76   |  42.86  
meta-llama/Llama-3-70b-chat-hf              |  37.50   |  47.02   |  34.44   |  39.51   |  39.62  
google/gemma-7b-it                          |  34.00   |  35.70   |  31.89   |  33.79   |  33.84  
reka-edge                                   |  31.84   |  39.40   |  30.02   |  32.36   |  33.41  
meta-llama/Llama-3-8b-chat-hf               |  28.91   |  34.19   |  31.52   |  31.79   |  31.60  
taide/Llama3-TAIDE-LX-8B-Chat-Alpha1        |  27.02   |  36.64   |  25.33   |  27.96   |  29.24  


# Installation

```bash
pip install git+https://github.com/theblackcat102/ievals.git
```

# Usage

```bash
ieval <model name> <series: optional> --top_k <numbers of incontext examples>
```

For more details please refer to [models](MODELS.md) section

# Coming soon

- Chain of Thought (CoT) with few shot

- Arxiv paper : detailed analysis on model interior and exterior relations

- More tasks

# Citation

```
@article{ikala2023eval,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting},
  journal={arXiv},
  year={2023}
}
```

## Disclaimer

This is not an officially supported iKala product.

This research code is provided "as-is" to the broader research community.
iKala does not promise to maintain or otherwise support this code in any way.
