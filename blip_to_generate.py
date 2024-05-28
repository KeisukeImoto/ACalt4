import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os
from tqdm import tqdm
import time
import csv
import json

#===============
# settings
source_dir = "xxx"
source_csv_dir = "xxxx.csv"

split_num = 0
seg = 0

target_csv_dir = f"./blipopt67b_audiocaps_seq{seg}_{split_num}.csv"

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

premodel_name = "opt6.7b"
if "opt" in premodel_name:
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type=f"pretrain_{premodel_name}", is_eval=True, device=device)
else:
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type=f"pretrain_{premodel_name}", is_eval=True, device=device)
#===============

   
rows = []
with open(source_csv_dir, "r", encoding="utf-8") as f:
    for iline, line in enumerate(f):
        rows.append(json.loads(line[:-1]))


if split_num == 0:
    seq_len = len(rows)
else:
    seq_len = len(rows) // split_num


for row in tqdm(rows[seq_len*seg:seq_len*(seg+1)]):
    jpg_files = [os.path.join(os.path.join(source_dir, row["dir"]), "frame/"+row["id"]+"."+frame) for frame in row["frame"]]
    out = []
    
    for jpg_file in jpg_files:
        raw_image = Image.open(jpg_file).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        out.append(model.generate({"image": image})[0])

    out_d= {"captions" : out}

    row.update(out_d)

    with open(target_csv_dir, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([json.dumps(row)])
