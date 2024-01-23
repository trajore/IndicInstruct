# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=1


# Evaluating Llama-2 7B base model using 0 shot
python -m eval.indicheadline.run_eval \
    --ntrain 0 \
    --save_dir results/translate_test/llama-2/indicheadline/llama-2-7b-chat-0shot \
    --model_name_or_path /data/jaygala/llama_ckpts/llama-v2-hf/llama-2-7b-chat \
    --tokenizer_name_or_path /data/jaygala/llama_ckpts/llama-v2-hf/llama-2-7b-chat \
    --eval_batch_size 2 \
    --use_chat_format \
     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating Llama-2 7B chat model using 0 shot
python -m eval.indicheadline.run_eval \
    --ntrain 0 \
    --save_dir results/translate_test/llama-2/indicheadline/llama-2-7b-0shot \
    --model_name_or_path /data/jaygala/llama_ckpts/llama-v2-hf/llama-2-7b \
    --tokenizer_name_or_path /data/jaygala/llama_ckpts/llama-v2-hf/llama-2-7b \
    --eval_batch_size 2
