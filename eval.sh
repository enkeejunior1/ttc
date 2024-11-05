# teacher eval (explicit CoT) 
# export FOLDER="data/gsm8k"
# export SAVE="train_models/gsm8k/gpt2/teacher/checkpoint_0"
# echo $SAVE
# mkdir -p $SAVE
# export TOKENIZERS_PARALLELISM=false 
# export CUDA_VISIBLE_DEVICES="2"
# python src/eval_gpt.py \
#     --test_path ${FOLDER}/test.txt \
#     --batch_size 1  \
#     --model_path $SAVE

for order in 10 ; do
export FOLDER="data/${order}_by_${order}_mult"
export SAVE="train_models/${order}_by_${order}_mult/gpt2/teacher/checkpoint_0"
echo $SAVE
mkdir -p $SAVE
export TOKENIZERS_PARALLELISM=false 
export CUDA_VISIBLE_DEVICES="1"
python src/eval_gpt.py \
    --test_path ${FOLDER}/test_bigbench.txt \
    --batch_size 1  \
    --max_new_tokens 1024  \
    --model_path $SAVE
done