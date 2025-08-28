unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST
export HF_HOME="/workspace_yuekai/ai-dynamo/cache"

bash launch/audio_agg.sh --model Qwen/Qwen2-Audio-7B-Instruct
# bash launch/agg.sh --model llava-hf/llava-1.5-7b-hf