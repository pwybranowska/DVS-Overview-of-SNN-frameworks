import torch

CONFIG = {'ckpt_path' : r'D:\AGH_zadania\sem8\DVS\project\DVS_project\snnTorch\snntorch_gen1_final.pt'}


checkpoint = torch.load(CONFIG["ckpt_path"])
state_dict = checkpoint["model"]

for i in range(1, 5):
    print(state_dict[f'block{i}.lif.beta'])