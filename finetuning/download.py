from huggingface_hub import hf_hub_download

# 다운로드
file_path = hf_hub_download(
    repo_id="liuhaotian/LLaVA-Pretrain",
    filename="blip_laion_cc_sbu_558k.json",
    repo_type="dataset",
    local_dir="./data"
)

# 이름 변경
import os
os.rename("./data/blip_laion_cc_sbu_558k.json", "./data/pretrain_558k.json")