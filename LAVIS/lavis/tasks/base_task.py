"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
import math

from torch.utils.data import DataLoader
from geomloss import SamplesLoss
import datetime
import lavis.common.dist_utils as dist_utils
import torch.nn.functional as F
import time


import sys

sys.path.append('../FairCLIP')
from src.modules import fair_vl_group_dataset_blip, endless_loader

sys.path.append('../src')
from fundus_dataloader import FUNDUS_Dataset
from lavis.processors.blip_processors import Blip2ImageTrainProcessor, BlipCaptionProcessor

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.loss_for_FairCLIP = SamplesLoss(loss="sinkhorn", p=2, blur=1e-4)

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            if name == 'fundus':
                dataset_dir = '/work/10523/bansa125/ls6/FairCLIP/FairVLMed'
                subset = 'train'
                vis_processor = Blip2ImageTrainProcessor(image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0)
                text_processor = BlipCaptionProcessor(prompt='', max_words=datasets_config["fundus"]["max_words"]) # default settings
                print(text_processor)
                fundus_dataset = FUNDUS_Dataset(dataset_dir, subset, vis_processor, text_processor, datasets_config["fundus"]["summary_type"])
                datasets[name] = {"train": fundus_dataset}
            else:
                dataset_config = datasets_config[name]

                builder = registry.get_builder_class(name)(dataset_config)
                dataset = builder.build_datasets()

                datasets[name] = dataset

        return datasets
    
    def build_group_datasets(self):
        groups_in_attrs = [3, 2, 2, 3]
        attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
        
        dataset_dir = '/work/10523/bansa125/ls6/FairCLIP/FairVLMed'
        
        preprocess = Blip2ImageTrainProcessor(image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0)
        
        group_dataloaders = []
        for i in range(groups_in_attrs[attr_to_idx["race"]]):
            tmp_dataset = fair_vl_group_dataset_blip(dataset_dir, preprocess, 
                    text_source='note', summarized_note_file="gpt4_summarized_notes.csv", 
                    attribute="race", thegroup=i)
            tmp_dataloader = DataLoader(tmp_dataset, batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=False)
            group_dataloaders.append(endless_loader(tmp_dataloader))
            
        print("GROUP DATALOADERS: ", group_dataloaders)
        return group_dataloaders

    def train_step(self, model, samples):
        output = model(samples)
        
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError
    
    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        fairclip=False,
        group_dataloaders=None,
        memory_module=None,
        mask=None,
        mean_epoch=0,
        std_epoch=0,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            fairclip=fairclip,
            group_dataloaders=group_dataloaders,
            memory_module=memory_module,
            mask=mask,
            mean_epoch=mean_epoch,
            std_epoch=std_epoch,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        fairclip=False,
        group_dataloaders=None,
        memory_module=None,
        mask=None,
        mean_epoch=0,
        std_epoch=0,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters
                
            if (mask[i] and epoch > 0):
                loss = loss * weight_estimation(loss, mean_epoch, std_epoch)
            
            memory_module[i, epoch] = loss
                
            similarity, _ = compute_sim_matrix_images(model, samples["image"], samples["text_input"])
            
            
            correlations_with_batch = similarity.diag().float()
                
            if fairclip:
                for x in group_dataloaders:
                    fair_samples = next(x)         
                    fair_samples = prepare_sample(fair_samples, cuda_enabled=cuda_enabled)
                    similarity, _ = compute_sim_matrix_images(model, fair_samples["image"], fair_samples["text_input"])
                    
                    correlations_with_group = similarity.diag().float()
                    correlations_with_group /= correlations_with_group.sum()
                
                    loss = loss + (1e-6 * self.loss_for_FairCLIP(correlations_with_batch[:,None], correlations_with_group[:,None]))
                                
            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file

def compute_sim_matrix_images(model, input_images, input_texts, **kwargs):

    logging.info("Computing features for evaluation...")

    text = input_texts
    text_ids = []
    text_embeds = []
    text_atts = []
    text_input = model.module.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(model.device)
    text_feat = model.module.forward_text(text_input)
    text_embed = F.normalize(model.module.text_proj(text_feat))
    text_embeds.append(text_embed)
    text_ids.append(text_input.input_ids)
    text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    image = input_images

    image = image.to(model.device)
    image_feat, vit_feat = model.module.forward_image(image)
    image_embed = model.module.vision_proj(image_feat)
    image_embed = F.normalize(image_embed, dim=-1)

    vit_feats.append(vit_feat.cpu())
    image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    return sims_matrix, sims_matrix

def weight_estimation(loss, mean, std):
    return (1/(std * math.sqrt(2 * math.pi))) * math.exp(-((loss - mean)**2)/(2 * std**2))