from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/llm_vo_hoang_nhat_khang/workspace/tijepa/beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset",
    tokenizer=tokenizer,
    annotation_data_path="/llm_vo_hoang_nhat_khang/workspace/tijepa/vqa_dataset/vqa",
)