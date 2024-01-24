export CUDA_VISIBLE_DEVICES=0


model_name_or_path="sarvamai/OpenHathi-7B-Hi-v0.1-Base"

echo "evaluating openhathi base on winogrande ..."

# zero-shot
python3 -m eval.winogrande.run_eval \
    --ntrain 0 \
    --save_dir "results/winogrande/openhathi-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8


model_name_or_path="ai4bharat/Airavatha"

echo "evaluating airavatha on winogrande ..."

# zero-shot
python3 -m eval.winogrande.run_eval \
    --ntrain 0 \
    --save_dir "results/winogrande/airavatha-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
