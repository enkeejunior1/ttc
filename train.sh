#########
# gsm8k #
#########
export FOLDER=data/gsm8k
export EPOCHS=1
export LR=5e-5
export BSZ=32

# MDLM
export MODEL=mdlm
export SAVE="train_models/gsm8k/$MODEL/teacher"
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0
python3 src/train_ddm.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --save_model $SAVE

# GPT2
export MODEL=gpt2
export SAVE="train_models/gsm8k/$MODEL/teacher"
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


# teacher (explicit CoT)
# for order in 10 8 6; do
# export FOLDER="data/${order}_by_${order}_mult"
# export MODEL=mdlm
# export EPOCHS=1
# export LR=5e-5
# export BSZ=32
# export SAVE="train_models/${order}_by_${order}_mult/mdlm/teacher"
# echo $SAVE
# mkdir -p $SAVE
# export TOKENIZERS_PARALLELISM=false 
# export CUDA_VISIBLE_DEVICES="0"
# python src/train_ddm.py \
#     --train_path ${FOLDER}/train.txt \
#     --val_path ${FOLDER}/valid.txt \
#     --epochs $EPOCHS    \
#     --lr $LR            \
#     --batch_size $BSZ   \
#     --base_model $MODEL \
#     --save_model $SAVE
# done

#######
# GPT #
#######
# teacher (explicit CoT)
# export FOLDER=data/gsm8k
# export MODEL=gpt2
# export EPOCHS=1
# export LR=5e-5
# export BSZ=32
# export SAVE=train_models/gsm8k/gpt2/teacher
# echo $SAVE
# mkdir -p $SAVE
# TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES=0
# python3 src/train_gpt.py \
#     --train_path ${FOLDER}/train.txt \
#     --val_path ${FOLDER}/valid.txt \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch_size $BSZ \
#     --base_model $MODEL \
#     --save_model $SAVE

# teacher (explicit CoT)
# for order in 10 8 6; do
# export FOLDER="data/${order}_by_${order}_mult"
# export MODEL=gpt2
# export EPOCHS=1
# export LR=5e-5
# export BSZ=32
# export SAVE="train_models/${order}_by_${order}_mult/gpt2/teacher"
# echo $SAVE
# mkdir -p $SAVE
# export TOKENIZERS_PARALLELISM=false 
# export CUDA_VISIBLE_DEVICES="0"
# python src/train_gpt.py \
#     --train_path ${FOLDER}/train.txt \
#     --val_path ${FOLDER}/valid.txt \
#     --epochs $EPOCHS    \
#     --lr $LR            \
#     --batch_size $BSZ   \
#     --base_model $MODEL \
#     --save_model $SAVE
# done