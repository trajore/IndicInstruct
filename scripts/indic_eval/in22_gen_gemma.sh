# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

echo "---------------------------------------------"
echo "             SINGLE TURN EVALUATION          "
echo "---------------------------------------------"


echo "---------------------------------------------"
echo "             SINGLE SYSTEM PROMPT            "
echo "---------------------------------------------"

model_name_or_path="google/gemma-7b"

echo "evaluating gemma 7b base on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4


model_name_or_path="google/gemma-7b-it"

echo "evaluating gemma 7b it on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format
    
    
echo "---------------------------------------------"
echo "             MULTI SYSTEM PROMPT             "
echo "---------------------------------------------"

model_name_or_path="google/gemma-7b"


model_name_or_path="google/gemma-7b-it"

echo "evaluating gemma 7b it on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format2

# 5-shot
python3 -m eval.in22.run_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format2
    
    
    
    
    
    
    
echo "---------------------------------------------"
echo "             MULTI TURN EVALUATION           "
echo "---------------------------------------------"


echo "---------------------------------------------"
echo "             SINGLE SYSTEM PROMPT            "
echo "---------------------------------------------"

model_name_or_path="google/gemma-7b"

echo "evaluating gemma 7b base on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8

# 5-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4


model_name_or_path="google/gemma-7b-it"

echo "evaluating gemma 7b it on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format

# 5-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format
    
    
echo "---------------------------------------------"
echo "             MULTI SYSTEM PROMPT             "
echo "---------------------------------------------"

model_name_or_path="google/gemma-7b"


model_name_or_path="google/gemma-7b-it"

echo "evaluating gemma 7b it on in22-gen ..."

# zero-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 0 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-0shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format2

# 5-shot
python3 -m eval.in22.run_multi_turn_eval \
    --ntrain 5 \
    --dataset "ai4bharat/IN22-Gen" \
    --src_lang eng_Latn --tgt_lang hin_Deva \
    --save_dir "results/in22-gen/gemma-7b-it-5shot" \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $model_name_or_path \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format2
    
    
    
# 5-shot
#python3 -m eval.in22.run_multi_turn_eval --ntrain 5 --dataset "ai4bharat/IN22-Gen" --src_lang eng_Latn --tgt_lang hin_Deva --save_dir "results/in22-gen/gemma-7b-it-5shot" --model_name_or_path $model_name_or_path --tokenizer_name_or_path $model_name_or_path --eval_batch_size 40 --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_gemma_chat_format2
