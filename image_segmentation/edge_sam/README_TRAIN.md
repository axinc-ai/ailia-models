## Table of Contents

- [Data Path](#data)
- [Prepare Teacher Embeddings](#teacher)
- [(Phase 1) Encoder-Only Knowledge Distillation](#encoder)
- [(Phase 2) Prompt-in-the-Loop Knowledge Distillation](#prompt)
- [Evaluation](#eval)

## Data Path <a name="data"></a>

Our model is trained on 1% data from the [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset. Please refer to [train subset](training/sa_train_subset.txt)/[val subset](training/sa_val_subset.txt) and organize the data as follows:

```
EdgeSAM
├── datasets
│   ├── SA-1B
│   │   ├── images
│   │   │   ├── train
│   │   │   │   ├── xxx.jpg
│   │   │   ├── val
│   │   │   │   ├── yyy.jpg
│   │   ├── annotations
│   │   │   ├── train
│   │   │   │   ├── xxx.json
│   │   │   ├── val
│   │   │   │   ├── yyy.json
```


## Prepare Teacher Embeddings <a name="teacher"></a>

1. Download the weights of the SAM ViT-H from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it at `weights/sam_vit_h_4b8939.pth`

2. Run the following commands to infer the teacher model (SAM ViT-H) and save the embedding at `teacher_embed/sa-1b/`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/save_embedding.py --cfg training/configs/teacher/sam_vit_huge_sa1b.yaml \
    --batch-size 8 \
    --eval \
    --resume weights/sam_vit_h_4b8939.pth
```

Note: adjust the number of GPUs and the batch size to fit your experiment environment.

## (Phase 1) Encoder-Only Knowledge Distillation <a name="encoder"></a>

Run the following commands to start encoder-only KD:

Download [rep_vit weight](https://github.com/THU-MIG/RepViT/releases).

```
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth
mv repvit_m0_9_distill_300e.pth weights/repvit_m1_distill_300.pth
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/train.py --cfg training/configs/rep_vit_m1_fuse_sa_distill.yaml \
    --output ./output/ \
    --batch-size 8 \
    --use-sync-bn
```

Combine the image-encoder-only model with the original SAM mask decoder and prompt encoder:

```
python scripts/convert_weights.py output/rep_vit_m1_fuse_sa_distill/default/ckpt_epoch_9.pth --encoder-only
```

## (Phase 2) Prompt-in-the-Loop Knowledge Distillation <a name="prompt"></a>

Run the following code to extract prompt encoder and decoder weights from sam_vit_h_4b8939.pth.

```
from edge_sam import sam_model_registry, SamPredictor

src_path = "weights/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=src_path)
torch.save(sam.prompt_encoder.state_dict(),'weights/sam_vit_h_prompt_encoder.pth')
torch.save(sam.mask_decoder.state_dict(),'weights/sam_vit_h_mask_decoder.pth')
```

Run the following commands to start prompt-in-the-loop KD:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/train.py --cfg training/configs/rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml \
    --output ./output/ \
    --batch-size 2
```

Run the following code to extract the checkpoints of EdgeSAM.

```
import torch
with open("./output/rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_4.pth", "rb") as f:
	state_dict = torch.load(f)

torch.save(state_dict["model"], "weights/trained_model.pth")
```

## Evaluation <a name="eval"></a>

Evaluation script is provided [here](scripts/eval_mIoU.sh).
