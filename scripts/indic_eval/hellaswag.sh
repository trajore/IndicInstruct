# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# -------------------------------------------------------------
#                       Hellaswag
# -------------------------------------------------------------

model_name_or_path="sarvamai/OpenHathi-7B-Hi-v0.1-Base"

echo "evaluating openhathi base on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --save_dir "results/hellaswag/openhathi-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --save_dir "results/hellaswag/openhathi-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --save_dir "results/hellaswag/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --save_dir "results/hellaswag/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# -------------------------------------------------------------
#                       Indic Hellaswag
# -------------------------------------------------------------

model_name_or_path="sarvamai/OpenHathi-7B-Hi-v0.1-Base"

echo "evaluating openhathi base on hellaswag-hi ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "results/hellaswag-hi/openhathi-base-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4

# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "results/hellaswag-hi/openhathi-base-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1


model_name_or_path="ai4bharat/airavata"

echo "evaluating airavata on hellaswag ..."

# zero-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 0 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "results/hellaswag-hi/airavata-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# 5-shot
python3 -m eval.hellaswag.run_eval \
    --ntrain 5 \
    --dataset "Thanmay/hellaswag-translated" \
    --save_dir "results/hellaswag-hi/airavata-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
