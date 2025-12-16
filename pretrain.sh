accelerate launch --num_processes 6 scripts/train/train_llm_pretrain.py \
  --train_data /home/ecs-user/nas_original_data/Emilia/emilia_audio_text_dataset \
  --model_ckpt pretrained_models/Fun-CosyVoice3-0.5B/llm_multihead.pt \
  --output_dir cv3_llm_multihead_pretrain \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --tokenizer_onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx \
  --resume_from_checkpoint cv3_llm_multihead_pretrain/checkpoint-10000 \
  --onnx_use_cuda \
  --save_steps 10000 \
  --save_total_limit 20 \