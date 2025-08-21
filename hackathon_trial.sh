#!/bin/bash

# Run the script
# For this project, run.py internally calls get_completion() and uses default gpt-4o model; if you want to force the env default, update code to pass model="hf:<MODEL_REPO/MODEL_NAME>".
python run.py --agent_num 2 --step_num 2 --model "hf:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

## Default model: gpt-4o from OpenAI
python run.py --agent_num 2 --step_num 2 

