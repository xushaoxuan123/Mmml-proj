import random
import os
import sys
# 获取当前文件所在目录的上级目录（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *
from collections import defaultdict
from tqdm import tqdm

caption_file = "/home/jiahao_zhao/多模态机器学习/期末大作业/test"
name_list = os.listdir(caption_file)
caption_file = "/home/jiahao_zhao/多模态机器学习/期末大作业/data/archive/captions.txt"
name_dict = defaultdict(list)
with open(caption_file, "r") as f:
    for line in f:
        line = line.strip()
        tmp_name = line[:line.find(",")]
        if tmp_name in name_list:
            name_dict[tmp_name].append(line[line.find(",")+1:])
print(len(name_dict))
print(name_dict[name_list[0]])

model_path='/home/jiahao_zhao/多模态机器学习/期末大作业/infinity_model/infinity_2b_reg.pth'
vae_path='/home/jiahao_zhao/多模态机器学习/期末大作业/infinity_model/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/home/jiahao_zhao/多模态机器学习/期末大作业/faln-t5-xl'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=False
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)


for name in tqdm(name_list):
    prompt = name_dict[name][0]
    cfg = 3
    tau = 0.5
    h_div_w = 1/1 # aspect ratio, height:width
    seed = random.randint(0, 10000)
    enable_positive_prompt=0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )

    args.save_file = f"/home/jiahao_zhao/多模态机器学习/期末大作业/res_of_inf/{name}"
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')