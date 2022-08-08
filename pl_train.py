"""
-*- coding: utf-8 -*-
Time    : 2022/7/29 20:31
Author  : Wang
"""
import argparse
import warnings
import time
import json
import torch
import torch.nn.functional as F
from typing import Union
import models
from os.path import join as path_join
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, seed_everything
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertModel, logging as transformer_log
from Dataset import my_dataset, get_batch
from loss import BCELoss, ATLoss
from untils import all_accuracy, get_logger
from transformers.optimization import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
transformer_log.set_verbosity_error()

LOSS_FN = {"ATL": ATLoss, "BCE": BCELoss}


class PlModel(pl.LightningModule):
    def __init__(self, args: Union[argparse.Namespace, argparse.ArgumentParser], steps):
        super(PlModel, self).__init__()
        self.args = args
        bert_name = f'bert-base-{args.bert_type}'
        bert = BertModel.from_pretrained(bert_name)
        self.model = models.my_model3(args, PTM=bert)
        self.loss_fn = LOSS_FN[args.loss_fn](args)
        self.loss_list = []
        self.acc = all_accuracy()
        self.total_step = steps
        self.save_hyperparameters(logger=True)

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        pred, loss = self.loss_fn(pred=output, batch=batch)
        self.compute_output(output=pred, batch=batch)
        self.loss_list.append(loss)
        log_dict = self.acc.get()
        log_dict['info'] = float(0)
        log_dict['loss'] = torch.stack(self.loss_list).mean()
        log_dict['lr'] = self.lr_schedulers().get_last_lr()[0]
        log_dict['epoch'] = float(self.current_epoch)
        self.log_dict(log_dict, prog_bar=False)
        return loss

    def training_epoch_end(self, outputs):
        self.acc.clear()
        self.loss_list = []

    def configure_optimizers(self):
        PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('PTM' in n)]
        not_PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('PTM' not in n)]
        optimizer = optim.AdamW([{'params': PLM, 'lr': self.args.pre_lr},
                                 {'params': not_PLM, 'lr': self.args.lr}])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(self.total_step * self.args.warm_ratio),
                                                    num_training_steps=self.total_step)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'step'
            }
        }

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.loss_fn.push_result(output, batch)
        pred, loss = self.loss_fn(pred=output, batch=batch)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        loss = sum(validation_step_outputs) / len(validation_step_outputs)
        dev_result = self.loss_fn.get_result()
        dev_result['loss'] = round(float(loss), 6)
        dev_result['info'] = float(1)
        self.log_dict(dev_result, prog_bar=False)
        return

    def compute_output(self, output, batch):
        with torch.no_grad():
            relations = batch['relations'].bool()
            relation_mask = batch['relation_mask'].unsqueeze(2)
            top_index = F.one_hot(torch.argmax(output, dim=-1),
                                  num_classes=self.args.relation_num).bool()
            result = torch.zeros_like(output, dtype=torch.bool)
            result[top_index] = True
            result = result & relations & relation_mask

            gold = relations.sum(0).sum(0)
            pred = result.sum(0).sum(0)
            self.acc.add_NA(num=gold[0], correct_num=pred[0])
            self.acc.add_not_NA(num=gold[1:].sum(), correct_num=pred[1:].sum())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Pl_Model")
        parser.add_argument("--loss_fn", type=str, default="BCE", choices=['BCE', "ATL"])
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--pre_lr", type=float, default=5e-5)
        parser.add_argument("--warm_ratio", type=float, default=0.06)
        parser.add_argument("--relation_num", type=int, default=97)
        parser.add_argument("--data_path", type=str, default='./data')
        parser.add_argument("--result_dir", type=str, default='./result')

        return parent_parser


class MyLogger(LightningLoggerBase):
    def __init__(self, args):
        super(MyLogger, self).__init__()
        self.base_log = get_logger(path_join(args.log_path, args.save_name), is_print=args.log_print)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Logger")
        parser.add_argument("--log_print", action='store_true')
        parser.add_argument("--log_path", type=str, default=r"./log/")
        return parent_parser


class DataModule(LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        train_data_path = f"./data/train_{args.bert_type}"
        valid_data_path = f'./data/dev_{args.bert_type}'
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--is_zip", action="store_true")

        return parent_parser


def main(args):
    seed_everything(args.seed)
    dm = DataModule(args)
    total_step = len(dm.train_dataloader()) * args.max_epochs // len(args.devices)
    strategy = 'ddp' if len(args.devices) > 1 else None

    model = PlModel(args, total_step)
    my_log = MyLogger(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="all_f1",
        mode="max",
        dirpath=args.checkpoint_dir,
        filename=args.save_name,
    )

    trainer = pl.Trainer.from_argparse_args(args=args, logger=my_log, callbacks=[checkpoint_callback],
                                            strategy=strategy)
    trainer.fit(model=model, datamodule=dm)
    # trainer.validate(model=model, datamodule=dm, ckpt_path=args.checkpoint_dir + '/ATLOP_cased_AdmW.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')

    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint')
    parser.add_argument("--save_name", type=str, default='test')
    parser.add_argument("--bert_type", type=str, choices=['cased', 'uncased'], default='uncased')
    parser = DataModule.add_model_specific_args(parser)
    parser = MyLogger.add_model_specific_args(parser)
    parser = PlModel.add_model_specific_args(parser)
    train_args = parser.parse_args()

    main(train_args)
