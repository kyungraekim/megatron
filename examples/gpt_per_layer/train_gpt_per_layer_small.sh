#!/bin/bash

# Small-scale GPT training with per-layer embedding for testing/development
# This is a simplified version for single-GPU or small multi-GPU setups

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Single GPU setup for testing
GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Paths (modify these for your setup)
CHECKPOINT_PATH="./checkpoints/gpt_per_layer_small"
TENSORBOARD_LOGS_PATH="./logs/gpt_per_layer_small"
VOCAB_FILE="./data/vocab/gpt2-vocab.json"
MERGE_FILE="./data/vocab/gpt2-merges.txt"
DATA_PATH="./data/processed/my_data_text_document"

# Per-layer embedding specific paths
PER_LAYER_VOCAB_FILE="./data/vocab/per_layer_vocab.json"
PER_LAYER_DATA_PATH="./data/processed/my_data_per_layer_document"

# Create directories
mkdir -p $(dirname $CHECKPOINT_PATH)
mkdir -p $TENSORBOARD_LOGS_PATH
mkdir -p ./data/vocab

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Small model for testing
GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)

# Per-layer embedding configuration
PER_LAYER_EMBEDDING_ARGS=(
    --use-per-layer-embedding
    --per-layer-vocab-size 5000                     # Smaller vocab for testing
    --per-layer-hidden-size 256                     # Smaller hidden size
    --per-layer-vocab-file $PER_LAYER_VOCAB_FILE
    --per-layer-embed-scale-factor 1.0
    --per-layer-combination-method "scaled_add"
    --per-layer-gate-activation "silu"
    --per-layer-projection-dropout 0.1
    --log-per-layer-stats
    --log-per-layer-gradients
    --per-layer-stats-interval 50
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 16                          # Small batch for testing
    --train-iters 1000                              # Short training for testing
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 1.0e-4                                     # Higher LR for faster convergence
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-warmup-fraction 0.01
    --lr-decay-iters 800
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1                  # No model parallelism for small setup
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --per-layer-data-path $PER_LAYER_DATA_PATH
    --split 949,50,1
    --dataloader-type "cyclic"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10                               # Frequent logging for testing
    --save-interval 500
    --eval-interval 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

echo "Starting GPT training with per-layer embedding..."
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Data path: $DATA_PATH"
echo "Per-layer data path: $PER_LAYER_DATA_PATH"
echo "Per-layer vocab: $PER_LAYER_VOCAB_FILE"

# Check if data files exist
if [ ! -f "$DATA_PATH.bin" ]; then
    echo "ERROR: Main data file not found: $DATA_PATH.bin"
    echo "Please run data preprocessing first:"
    echo "python tools/preprocess_data.py --input your_data.json --output-prefix ./data/processed/my_data --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --tokenizer-type GPT2BPETokenizer"
    exit 1
fi

if [ ! -f "$PER_LAYER_DATA_PATH.bin" ]; then
    echo "ERROR: Per-layer data file not found: $PER_LAYER_DATA_PATH.bin"
    echo "Please run per-layer data preprocessing first:"
    echo "python preprocess_per_layer_data.py --input your_data.json --output-prefix ./data/processed/my_data --per-layer-output-prefix ./data/processed/my_data_per_layer --vocab-file $VOCAB_FILE --per-layer-vocab-file $PER_LAYER_VOCAB_FILE"
    exit 1
fi

# Run training
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt_per_layer.py \
    ${GPT_MODEL_ARGS[@]} \
    ${PER_LAYER_EMBEDDING_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

echo "Training completed!"
echo "Check logs at: $TENSORBOARD_LOGS_PATH"
echo "Checkpoints saved to: $CHECKPOINT_PATH"