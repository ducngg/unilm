python -m torch.distributed.launch --nproc_per_node=1 eval.py \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 1 \
        --sentencepiece_model /llm_vo_hoang_nhat_khang/workspace/tijepa/beit3.spm \
        --finetune /llm_vo_hoang_nhat_khang/workspace/tijepa/beit3_base_patch16_480_vqa.pth \
        --data_path /llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset \
        --output_dir /llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset/pred \
        --eval 
        # \
        # --dist_eval