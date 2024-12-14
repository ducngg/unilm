import torch
# print(torch.version.cuda)

import sys
from argparse import Namespace
import pickle

# from torchscale.architecture.utils import init_bert_params
# from torchscale.component.droppath import DropPath
# from torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
# from torchscale.component.multihead_attention import MultiheadAttention
# from torchscale.component.multiway_network import MultiwayWrapper, set_split_position
# from torchscale.component.relative_position_bias import RelativePositionBias
# from torchscale.component.xmoe.moe_layer import MOELayer
# from torchscale.component.xmoe.routing import Top1Gate, Top2Gate


# from torchscale.architecture.encoder import Encoder
# from torchscale.component.embedding import (
#     PositionalEmbedding,
#     TextEmbedding,
#     VisionEmbedding,
# )
# from torchscale.component.multiway_network import MutliwayEmbedding


# from torchscale.model.BEiT3 import BEiT3
from modeling_finetune import beit3_base_patch16_480_vqav2
# from run_beit3_finetuning import get_args

# args = get_args()[0]

# beit_train_transform = build_transform(is_train=True, args=args)
# beit_eval_transform = build_transform(is_train=False, args=args)

# beit_args = _get_base_config(**vars(args))
# beit_wrapper = BEiT3BinaryImageClassification(beit_args)
# utils.load_model_and_may_interpolate(args.finetune, beit_wrapper, args.model_key, args.model_prefix)
# beit_wrapper.to(args.device)
# beit_wrapper.beit3.eval()


"""
"""

device = 'cuda:0'

model = beit3_base_patch16_480_vqav2()

ckp = torch.load('/llm_vo_hoang_nhat_khang/workspace/tijepa/beit3_base_indomain_patch16_480_vqa.pth', map_location="cpu")

model_state_dict = ckp['model']

model.load_state_dict(model_state_dict, strict=False)

model.to(device).eval()



from run_beit3_finetuning import get_args

# python -m torch.distributed.launch --nproc_per_node=1 eval.py \
#         --model beit3_base_patch16_480 \
#         --input_size 480 \
#         --task vqav2 \
#         --batch_size 1 \
#         --sentencepiece_model /llm_vo_hoang_nhat_khang/workspace/tijepa/beit3.spm \
#         --finetune /llm_vo_hoang_nhat_khang/workspace/tijepa/beit3_base_patch16_480_vqa.pth \
#         --data_path /llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset \
#         --output_dir /llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset/pred \
#         --eval 
#         # \
#         # --dist_eval
sys.argv = [
    'eval.py',
    '--model', 'beit3_base_patch16_480',
    '--input_size', '480',
    '--task', 'vqav2',  # just because required
    '--batch_size', '100',
    '--sentencepiece_model', '/llm_vo_hoang_nhat_khang/workspace/tijepa/beit3.spm',
    '--finetune', '/llm_vo_hoang_nhat_khang/workspace/tijepa/beit3_base_patch16_480_vqa.pth',
    '--data_path', '/llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset',
    '--output_dir', '/llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset/pred_test',
    '--num_workers', '0',
    '--eval'
    # '--randaug',  # augmentations
]
args = get_args()[0]

from datasets import create_downstream_dataset
data_loader_test = create_downstream_dataset(args, is_eval=True)

from engine_for_finetuning import get_handler, evaluate
task_handler = get_handler(args)

print('Evaluating...')
# result, _ = evaluate(data_loader_test, model, device, task_handler)
print('Done evaluate...')

# with open('RESULT.pkl', 'wb') as f:
#     pickle.dump(result, f)

with open('RESULT.pkl', 'rb') as f:
    result = pickle.load(f)

# print(result)
jsons = result

try:
    import json
    result_file = os.path.join(
        '/llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset/pred_test',  
        f"submit.json"
    )
    with open(result_file, "w") as fp:
        json.dump(jsons, fp, indent=2)
        print("Infer %d examples into %s" % (len(result), result_file))

except:
    pass


import utils
utils.dump_predictions(args, result, "vqav2_test")



