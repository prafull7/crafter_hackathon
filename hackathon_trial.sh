#!/bin/bash

export HF_HOME="<YOUR_HF_HOME>"
export HF_DATASETS_CACHE="<YOUR_HF_DATASETS_CACHE>"

export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export OPENAI_CHAT_MODEL="<YOUR_OPENAI_CHAT_MODEL>"

# For HF models, pass model="hf:<MODEL_REPO/MODEL_NAME>".
python run.py --agent_num 2 --step_num 2 --model "hf:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

## Default model: gpt-4o from OpenAI
python run.py --agent_num 2 --step_num 2 

