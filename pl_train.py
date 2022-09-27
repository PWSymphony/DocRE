"""
-*- coding: utf-8 -*-
Time    : 2022/7/29 20:31
Author  : Wang
"""
import argparse
import json
import logging
import platform
import time
import warnings
from os.path import join as path_join
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertModel, logging as transformer_log
from transformers.optimization import get_linear_schedule_with_warmup

import models
from Dataset import my_dataset, get_batch
from loss import BCELoss, ATLoss, MultiLoss, CELoss
from untils import all_accuracy, get_logger, Accuracy

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
transformer_log.set_verbosity_error()

LOSS_FN = {"ATL": ATLoss, "BCE": BCELoss, "Multi": MultiLoss}
MODELS = {'model': models.my_model, 'model1': models.my_model1, 'model2': models.my_model2,
          'model3': models.my_model3, 'model4': models.my_model4, 'model5': models.my_model5}


class PlModel(pl.LightningModule):
    def __init__(self, args: Union[argparse.Namespace, argparse.ArgumentParser]):
        super(PlModel, self).__init__()
        self.args = args
        bert_name = f'bert-base-{args.bert_type}'
        bert = BertModel.from_pretrained(bert_name)
        self.model = MODELS[args.model](args, bert)
        self.loss_fn = LOSS_FN[args.loss_fn](args)
        self.loss_fn_2 = CELoss(args)
        self.loss_list = []
        self.acc = all_accuracy()
        self.save_hyperparameters(logger=True)

        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        bin_res, relation_res = self.model(batch, is_train=True)
        _, loss1 = self.loss_fn_2(bin_res, **batch)
        pred, loss2 = self.loss_fn(pred=relation_res, **batch)

        self.compute_output(output=pred, batch=batch)
        self.loss_list.append(loss1 + loss2)

        log_dict = self.acc.get()
        log_dict['info'] = float(0)
        log_dict['loss'] = torch.stack(self.loss_list).mean()
        log_dict['lr'] = self.lr_schedulers().get_last_lr()[0]
        log_dict['epoch'] = float(self.current_epoch)
        self.log_dict(log_dict, prog_bar=False)

        return loss1 + loss2

    def training_epoch_end(self, outputs):
        self.acc.clear()
        self.loss_list.clear()

    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [{
            "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay, "lr": self.args.lr},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": self.args.pre_lr}]
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(self.args.total_step * self.args.warm_ratio),
                                                    num_training_steps=self.args.total_step)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": 'step'}}

    def validation_step(self, batch, batch_idx):
        bin_res, relation_res = self.model(batch)
        self.loss_fn.push_result(relation_res, batch)
        pred, loss = self.loss_fn(pred=relation_res, **batch)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        dev_result = self.loss_fn.get_result()
        dev_result['info'] = float(1)
        self.log_dict(dev_result, prog_bar=False)
        return

    def compute_output(self, output, batch):
        with torch.no_grad():
            relations = batch['relations'].bool()
            relation_mask = batch['relation_mask'].unsqueeze(2)
            top_index = F.one_hot(torch.argmax(output, dim=-1),
                                  num_classes=self.args.relation_num).bool()
            result = top_index & relations & relation_mask

            # gold.sum(0).sum(0) 一对实体有多种关系会被计算为多对实体
            gold_na = relations[..., 0].sum()
            gold_not_na = relations[..., 1:].nonzero().shape[0]  # 先求和，在将不为0的改为1，再次求和
            pred_na = result[..., 0].sum()
            pred_not_na = result[..., 1:].sum()
            self.acc.add_NA(num=gold_na, correct_num=pred_na)
            self.acc.add_not_NA(num=gold_not_na, correct_num=pred_not_na)


class MyLogger(LightningLoggerBase):
    def __init__(self, args):
        super(MyLogger, self).__init__()
        log_path = path_join(args.log_path, args.save_name)
        pl_logger = logging.getLogger('pytorch_lightning')
        pl_logger.addHandler(logging.FileHandler(log_path + '.txt'))

        self.base_log = get_logger(log_path, is_print=args.print_log)
        cur_time = time.strftime('%Y年%m月%d日, %H:%M', time.localtime())
        self.base_log.info(cur_time)

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return time.strftime('%m.%d_%H:%M')

    @rank_zero_only
    def log_hyperparams(self, params):
        out = {}
        for k in params.keys():
            if isinstance(params[k], argparse.Namespace):
                out.update(params[k].__dict__)
            else:
                out[k] = params[k]
        out = json.dumps(out, indent=4)
        self.base_log.info(out)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if 'info' in metrics and metrics['info'] == 0:
            m = f"step:{int(step) + 1:5d} | epoch:{int(metrics['epoch'] + 1):2d} | loss:{metrics['loss']:.6f} | " \
                f"NA:{metrics['NA']:2.2f} | " \
                f"not NA:{metrics['not_NA']:2.2f} | total:{metrics['total']:2.2f} | lr:{metrics['lr']:.2e}"
        elif 'info' in metrics and metrics['info'] == 1:
            m = f"  all_F1:{metrics['all_f1'] * 100:2.2f}  |  ign_F1:{metrics['ign_f1'] * 100:2.2f}  |  " \
                f"ign_theta_f1: {metrics['ign_theta_f1'] * 100: 2.2f}"
        else:
            m = str(metrics)
        self.base_log.info(m)


class DataModule(LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.batch_size = args.batch_size
        self.num_workers = 4 if args.accelerator == 'gpu' and platform.system() == 'Linux' else 0

        train_data_path = f'{args.data_path}/train_{args.bert_type}'
        valid_data_path = f'{args.data_path}/dev_{args.bert_type}'
        self.train_dataset = my_dataset(train_data_path, is_zip=args.is_zip)
        self.val_dataset = my_dataset(valid_data_path, is_zip=args.is_zip)

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, num_workers=self.num_workers,
                                      batch_size=self.batch_size, collate_fn=get_batch,
                                      pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size * 2, collate_fn=get_batch,
                                    num_workers=self.num_workers)
        return val_dataloader


def main(args):
    # ========================================== 检查参数 ==========================================
    if not torch.cuda.is_available():
        args.accelerator = 'cpu'

    # ========================================== 获取数据 ==========================================
    dm = DataModule(args)

    # ========================================== 配置参数 ==========================================
    seed_everything(args.seed)
    total_step = (len(dm.train_dataloader()) * args.max_epochs)
    strategy = None
    if args.accelerator == 'cpu':
        args.devices = None
        args.precision = 32
    elif args.accelerator == 'gpu' and len(args.devices) > 1:
        total_step //= len(args.devices)
        strategy = 'ddp'

    args.total_step = total_step

    my_log = MyLogger(args)
    callbacks = []
    if args.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="all_f1", mode="max", dirpath=args.checkpoint_dir,
                                              filename=args.save_name)
        callbacks.append(checkpoint_callback)

    # ========================================== 开始训练 ==========================================
    model = PlModel(args)
    trainer = pl.Trainer.from_argparse_args(args=args, logger=my_log, callbacks=callbacks, strategy=strategy)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')
    parser.add_argument("--gradient_clip_val", type=int, default=1)

    parser.add_argument("--bert_type", type=str, choices=['cased', 'uncased'], default='uncased')
    parser.add_argument("--loss_fn", type=str, default="ATL")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pre_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warm_ratio", type=float, default=0.06)
    parser.add_argument("--relation_num", type=int, default=97)
    parser.add_argument("--result_dir", type=str, default='./result')
    parser.add_argument("--model", type=str, default='model')

    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--total_step", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--is_zip", action="store_true")

    parser.add_argument("--print_log", action='store_true')
    parser.add_argument("--log_path", type=str, default=r"./log/")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint')
    parser.add_argument("--save_name", type=str, default='test')

    train_args = parser.parse_args()

    main(train_args)
