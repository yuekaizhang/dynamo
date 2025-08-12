

cd $DYNAMO_HOME/components/backends/trtllm
# huggingface-cli download --local-dir Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-7B-Instruct
export MODEL_PATH=${MODEL_PATH:-"/home/scratch.yuekaiz_wwfo_1/ai-dynamo/Qwen2-VL-7B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen2-VL-7B-Instruct"}

export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"decode_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"engine_configs/multimodal/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"engine_configs/multimodal/decode.yaml"}

export MODALITY=${MODALITY:-"multimodal"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"engine_configs/multimodal/agg.yaml"}

#./launch/disagg.sh
./launch/agg.sh