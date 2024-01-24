export CUDA_VISIBLE_DEVICES=0


model_name_or_path="sarvamai/OpenHathi-7B-Hi-v0.1-Base"

echo "evaluating openhathi base on indicxnli ..."

# zero-shot
python3 -m eval.indicxnli.run_eval \
    --ntrain 0 \
    --save_dir "results/indicxnli/openhathi-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.indicxnli.run_eval \
    --ntrain 5 \
    --save_dir "results/indicxnli/openhathi-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on indicxnli ..."

# zero-shot
python3 -m eval.indicxnli.run_eval \
    --ntrain 0 \
    --save_dir "results/indicxnli/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.indicxnli.run_eval \
    --ntrain 5 \
    --save_dir "results/indicxnli/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
