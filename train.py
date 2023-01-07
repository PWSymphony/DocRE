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
from transformers import logging as transformer_log
from transformers.optimization import get_linear_schedule_with_warmup

from Dataset import DataModule
from loss import ATLoss, BCELoss
from models import ReModel
from process_data import Processor
from utils import AllAccuracy, MyLogger

warnings.filterwarnings("ignore", category=UserWarning)
transformer_log.set_verbosity_error()

LOSS_FN = {"ATL": ATLoss, "BCE": BCELoss}


class PlModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(PlModel, self).__init__()
        self.args = args
        self.model = ReModel(args)
        self.loss_fn = LOSS_FN[args.loss_fn](args)
        self.loss_list = []
        self.all_acc = AllAccuracy()
        self.save_hyperparameters(logger=True)
        self.max_f1 = -1

    def configure_optimizers(self):
        total_step = self.args.total_step
        plm = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' in n)]
        not_plm = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' not in n)]
        optimizer = optim.AdamW([{'params': plm, 'lr': self.args.pre_lr},
                                 {'params': not_plm, 'lr': self.args.lr}])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(total_step * self.args.warm_ratio),
                                                    num_training_steps=total_step)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": 'step'}}

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        pred = output.get('pred', None)
        loss = self.loss_fn(pred=pred, batch=batch)
        self.compute_output(output=pred, label=batch['relations'], mask=batch['relation_mask'], )

        self.loss_list.append(loss)
        log_dict = {}
        log_dict.update(self.all_acc.get())
        log_dict['loss'] = torch.stack(self.loss_list).mean()
        log_dict['lr'] = self.lr_schedulers().get_last_lr()[0]
        log_dict['epoch'] = float(self.current_epoch)
        self.log_dict(log_dict, prog_bar=False)
        return loss

    def training_epoch_end(self, outputs):
        self.all_acc.clear()
        self.loss_list = []
        self.log_dict({'epoch': -1}, prog_bar=False)

    def on_validation_start(self):
        self.all_acc.clear()
        self.loss_list = []

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        pred = output.get('pred', None)
        self.loss_fn.push_result(pred, batch)
        loss = self.loss_fn(pred=pred, batch=batch)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        dev_result = self.loss_fn.get_result()
        self.max_f1 = max(self.max_f1, dev_result.get('all_f1', -1))
        dev_result['max f1'] = self.max_f1
        self.log_dict(dev_result, prog_bar=False)

    def test_step(self, batch, batch_idx):
        output, _ = self.model(**batch)
        self.loss_fn.push_result(output, batch)

    def test_epoch_end(self, outputs):
        self.loss_fn.get_result(is_test=True)

    def compute_output(self, output, label, mask=None):
        with torch.no_grad():
            cur_label = label.bool()
            top_index = F.one_hot(torch.argmax(output, dim=-1),
                                  num_classes=output.shape[-1]).bool()
            result = top_index & cur_label
            if mask is not None:
                if len(mask.shape) < len(cur_label.shape):
                    cur_mask = mask.unsqueeze(2)
                result = result & cur_mask
            # gold.sum(0).sum(0) 一对实体有多种关系会被计算为多对实体
            gold_na = cur_label[..., 0].sum()
            gold_not_na = cur_label[..., 1:].sum(-1).bool().sum()  # 先求和，在将不为0的改为1，再次求和
            pred_na = result[..., 0].sum()
            pred_not_na = result[..., 1:].sum()
            self.all_acc.add_NA(num=gold_na, correct_num=pred_na)
            self.all_acc.add_not_NA(num=gold_not_na, correct_num=pred_not_na)


def main(args):
    # ========================================== 处理数据 ==========================================
    if args.process_data:
        processor = Processor(args)
        processor()
        print("数据处完成")
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
        checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                              save_weights_only=True,
                                              monitor="all_f1",
                                              mode="max",
                                              dirpath=args.checkpoint_dir,
                                              filename="" + args.save_name + "-{all_f1}")
        callbacks.append(checkpoint_callback)

    # ========================================== 开始训练 ==========================================
    trainer = pl.Trainer.from_argparse_args(args=args, logger=my_log, callbacks=callbacks, strategy=strategy,
                                            num_sanity_val_steps=0)

    if args.is_test:
        ckpt = os.path.join(args.checkpoint_dir, args.save_name + '.ckpt')
        model = PlModel.load_from_checkpoint(ckpt)
        trainer.test(model=model, datamodule=datamodule)
    else:
        model = PlModel(args)
        trainer.fit(model=model, datamodule=datamodule)
        # model = model.load_from_checkpoint(r'E:\pythonProject\MyDocRE\checkpoint\ATLOP.ckpt')
        # trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')

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
    parser.add_argument("--is_test", action='store_true')

    parser.add_argument("--hidden_log", action='store_true')
    parser.add_argument("--log_path", type=str, default=r"./log/")

    main(parser.parse_args())
