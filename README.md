
# Robust Fairness Vision-Language Learning for Medical Image Analysis
<!-- > [**CVPR 2024**] [**FairCLIP: Harnessing Fairness in Vision-Language Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.pdf) -->
>
> by Sparsh Bansal*, Mingyang Wu*, Xin Wang*, Shu Hu*
>


## Abstract

The advent of Vision-Language Models (VLMs)
in medical image analysis has the potential to help process
multimodal inputs and increase performance over traditional
inference methods. However, when considering the domain in
which these models will be implemented, fairness and robustness
are important to ensure the model stays true for any patient. In
this paper, we introduce a framework for ensuring robustness
and fairness of VLM models. This framework modifies the loss
function at training by identifying and adjusting faulty image-
text pairs through a Dynamic Bad Pair Mining algorithm and
also utilizing Sinkhorn distance to ensure the loss distributions of
protected groups do not deviate from the total loss. Experimental
testing of our framework shows up to a 8.6% improvement when
looking at equity-scaled AUC. 


## Installation

To set up the required environment:

```bash
conda env create -f robust_fairclip.yml
```

## Dataset
The Harvard-FairVLMed dataset from [FairCLIP](https://arxiv.org/pdf/2403.19949) can be accessed via this [link](https://drive.google.com/drive/folders/1bkeifigwOAfnsLvup9mJOSNeA3WsvA2l?usp=drive_link). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.

Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

The Harvard-FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
Harvard-FairVLMed
├── data_summary.csv
├── gpt-4_summarized_notes.csv
├── Training
├── Validation
└── Test
```
The file split_files.csv details the division of data into training, validation, and testing sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxx.npz", where "xxxxx" (e.g., 06691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Moreover, the files original_notes.csv and gpt-4_summarized_notes.csv contain original notes and notes summarized by GPT-4, respectively.

Each NPZ file has the following fields.
```
slo_fundus: slo fundus image
md: visual field mean deviation
tds: 52 visual field total deviation values
age: patient age
gender: Female (0), Male (1)
race: Asian (0), Black (1), White (2)
ethnicity: non-Hispanic (0), Hispanic (1), Unknown (-1)
language: English (0), Spanish (1), Other (2), Unknown (-1)
maritalstatus: Marriage or Partnered (0), Single (1), Divorced (2), Widoled (3), Legally Separated (4), Unknown (-1)
glaucoma: Non-Glaucoma (0) or Glaucoma (1)
note: the original de-identified clinical note
note_extra: the original de-identified clinical note with demographic attributes placed at the beginning
```

## LLM Summarization
We use the following LLMs for summarizing the medical notes.
1. PMC-LLAMA
2. MED42
3. GPT-4

```bash
python src/dataset_deidentification_summarization.py --openai_key <YOUR_OPENAI_KEY> --models gpt-4
```

NOTE: OPENAI_KEY is only needed for GPT-4.

## Pre-training

### RobustFairCLIP
The code for pre-training **CLIP**, **FairCLIP**, and **RobustFairCLIP** is in the folder [FairCLIP](./FairCLIP).

### RobustBLIP-2
```bash
cd Robust_Fairness_For_Medical_Image/LAVIS
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
```

## Evaluation

### Linear Probing
```bash
cd Robust_Fairness_For_Medical_Image/mae
DATA_DIR=/Path/to/FairVLMed
FEATS_TYPE=image # [image, multimodal]

PRETRAIN_CHKPT=/Path/to/CKPT
EXP_NAME=tmp
MODEL_TYPE=blip2 # [clip, blip2]

OMP_NUM_THREADS=1 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=1 main_linprobe.py --model_type ${MODEL_TYPE} --vl_feats_type ${FEATS_TYPE} --blip_feats_select avgpool --cfg-path ../LAVIS/lavis/projects/blip2/train/pretrain_stage1.yaml --vision_encoder_weights clip --summary_type gpt-4 --batch_size 512 --model vit_large_patch16 --cls_token --finetune ${PRETRAIN_CHKPT} --epochs 1000 --blr 0.1 --weight_decay 0.0 --data_path ${DATA_DIR} --output_dir $EXP_NAME --log_dir $EXP_NAME --nb_classes 2 > ${EXP_NAME}.out
```

## Acknowledgment

We acknowledge that part of our code is adapted from [FairCLIP](https://arxiv.org/pdf/2403.19949) (CVPR 2024), if you cite our paper, please consider citing their paper as well:

```bibtex
@inproceedings{luo2024fairclip,
  title={Fairclip: Harnessing fairness in vision-language learning},
  author={Luo, Yan and Shi, Min and Khan, Muhammad Osama and Afzal, Muhammad Muneeb and Huang, Hao and Yuan, Shuaihang and Tian, Yu and Song, Luo and Kouhana, Ava and Elze, Tobias and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12289--12301},
  year={2024}
}

```
