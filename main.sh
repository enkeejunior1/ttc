# teacher (explicit CoT)
export FOLDER=data/gsm8k
export MODEL=gpt2
export EPOCHS=1
export LR=5e-5
export BSZ=32
export SAVE=train_models/gsm8k/gpt2/teacher
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=1
python3 src/train_gpt.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --save_model $SAVE

# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
#     stdbuf -oL -eL python src/train_gpt.py \
#     --train_path ${FOLDER}/train.txt \
#     --val_path ${FOLDER}/valid.txt \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch_size $BSZ \
#     --base_model $MODEL \
#     --save_model $SAVE \
#     > ${SAVE}/log.train 2>&1&