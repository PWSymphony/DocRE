import argparse
import os
import platform
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from transformers import AutoTokenizer, AutoModel, logging as transformer_log
from transformers.optimization import get_linear_schedule_with_warmup

from models import ReModel
from loss import BCELoss, ATLoss
from utils import all_accuracy, Accuracy, MyLogger
from Dataset import DataModule
from process_data import Processor

warnings.filterwarnings("ignore", category=UserWarning)
transformer_log.set_verbosity_error()

LOSS_FN = {"ATL": ATLoss, "BCE": BCELoss}


class PlModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(PlModel, self).__init__()
        self.args = args
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        if 'roberta' in args.bert_name:
            cls_token_id = [tokenizer.cls_token_id]
            sep_token_id = [tokenizer.sep_token_id, tokenizer.sep_token_id]
        else:
            cls_token_id = [tokenizer.cls_token_id]
            sep_token_id = [tokenizer.sep_token_id]
        self.model = ReModel(args, AutoModel.from_pretrained(args.bert_name), cls_token_id, sep_token_id)
        self.loss_fn = LOSS_FN[args.loss_fn](args)
        self.loss_list = []
        self.acc = all_accuracy()
        self.save_hyperparameters(logger=True)

        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

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
        total_step = self.args.total_step
        PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' in n)]
        not_PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' not in n)]
        optimizer = optim.AdamW([{'params': PLM, 'lr': self.args.pre_lr},
                                 {'params': not_PLM, 'lr': self.args.lr}])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(total_step * self.args.warm_ratio),
                                                    num_training_steps=total_step)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": 'step'}}

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

    def compute_output(self, output, batch):
        with torch.no_grad():
            relations = batch['relations'].bool()
            relation_mask = batch['relation_mask'].unsqueeze(2)
            top_index = F.one_hot(torch.argmax(output, dim=-1),
                                  num_classes=self.args.relation_num).bool()
            result = top_index & relations & relation_mask

            # gold.sum(0).sum(0) 一对实体有多种关系会被计算为多对实体
            gold_na = relations[..., 0].sum()
            gold_not_na = relations[..., 1:].sum(-1).bool().sum()  # 先求和，在将不为0的改为1，再次求和
            pred_na = result[..., 0].sum()
            pred_not_na = result[..., 1:].sum()
            self.acc.add_NA(num=gold_na, correct_num=pred_na)
            self.acc.add_not_NA(num=gold_not_na, correct_num=pred_not_na)


def main(args):
    # ========================================== 处理数据 ==========================================
    if args.process_data:
        processor = Processor(args)
        processor()
        for path in os.listdir(args.data_path):
            if '.data' in path:
                os.remove(os.path.join(args.data_path, path))
        exit()

    # ========================================== 检查参数 ==========================================
    if not torch.cuda.is_available():
        args.accelerator = 'cpu'

    # ========================================== 获取数据 ==========================================
    datamodule = DataModule(args)

    # ========================================== 配置参数 ==========================================
    seed_everything(args.seed)
    total_step = (len(datamodule.train_dataloader()) * args.max_epochs)
    strategy = None
    if args.accelerator == 'cpu':
        args.devices = None
        args.precision = 32
    elif args.accelerator == 'gpu' and len(args.devices) > 1:
        total_step //= len(args.devices)
        strategy = 'ddp'

    if platform.system().lower() != 'linux':
        args.num_workers = 0

    args.total_step = total_step

    my_log = MyLogger(args)
    callbacks = []
    if args.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="all_f1", mode="max", dirpath=args.checkpoint_dir,
                                              filename=args.save_name)
        callbacks.append(checkpoint_callback)

    # ========================================== 开始训练 ==========================================
    model = PlModel(args)
    trainer = pl.Trainer.from_argparse_args(args=args, logger=my_log, callbacks=callbacks, strategy=strategy,
                                            num_sanity_val_steps=0)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    parser.add_argument("--loss_fn", type=str, default="ATL", choices=['BCE', "ATL"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pre_lr", type=float, default=5e-5)
    parser.add_argument("--warm_ratio", type=float, default=0.06)
    parser.add_argument("--relation_num", type=int, default=97)
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument("--result_dir", type=str, default='./result')

    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint')
    parser.add_argument("--save_name", type=str, default='test')
    parser.add_argument("--bert_name", type=str, default='bert-base-uncased')

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--raw_path", type=str, default="./data/raw")
    parser.add_argument("--meta_path", type=str, default="./data/meta")
    parser.add_argument("--data_type", type=str, default="", choices=['', 'revised'])
    parser.add_argument("--process_data", action='store_true')

    parser.add_argument("--hidden_log", action='store_true')
    parser.add_argument("--log_path", type=str, default=r"./log/")

    main(parser.parse_args())
