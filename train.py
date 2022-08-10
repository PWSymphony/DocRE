import os
from os.path import join as opj
import time
import models
import torch
import numpy as np
from math import ceil
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import seed_everything
from transformers import logging as log
from transformers import BertModel
from Dataset import my_dataset, get_batch
from untils import Accuracy, log_config_information, get_params, get_logger
from test import test, ATLtest
from Config import Config, option
import platform
from loss import ATLoss, BCELoss
from transformers.optimization import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

load_data_works = 1
if platform.system().lower() == 'linux':
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')
    load_data_works = 3

log.set_verbosity_error()

loss_fn_dict = {'BCE': BCELoss, 'ATL': ATLoss}
test_fn_dict = {'BCE': test, 'ATL': ATLtest}


class trainer:
    def __init__(self, config: Config, model: nn.Module, Logger=None):
        seed_everything(config.seed)

        param_nums = get_params(model)
        self.config = config
        self.logger = Logger
        self.logger.info(param_nums)

        train_data = my_dataset(opj(config.data_path, f'{config.train_prefix}_{config.data_type}'))
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        if config.use_gpu:
            model.cuda()
        else:
            model.cpu()
        self.model = model

        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data,
                                           sampler=train_sampler,
                                           batch_size=config.batch_size,
                                           collate_fn=get_batch,
                                           num_workers=load_data_works,
                                           prefetch_factor=4)

        self.total_step = len(self.train_dataloader) * config.epochs
        self.evaluate_epoch = config.evaluate_epoch
        self.warm_step = config.warm_ratio * self.total_step
        self.decay_step = config.decay_ratio * self.total_step
        self.logger.info(f"total step:{self.total_step}  evaluate epoch:{self.evaluate_epoch}")

        PLM = [p for n, p in self.model.named_parameters() if p.requires_grad and ('PTM' in n)]
        not_PLM = [p for n, p in self.model.named_parameters() if p.requires_grad and ('PTM' not in n)]
        self.optimizer = optim.AdamW([{'params': PLM, 'lr': config.pre_lr},
                                     {'params': not_PLM, 'lr': config.lr}])
        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self.LR_Lambda)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_step, self.total_step)
        self.time = time.time()

        self.loss_fn = loss_fn_dict[con.loss_fn](config)
        self.test_fn = test_fn_dict[con.loss_fn]

    def LR_Lambda(self, step):
        if step < self.warm_step:
            return step / self.warm_step
        elif step > (self.total_step - self.decay_step):
            return np.power(0.9999, step - self.total_step + self.decay_step)
        else:
            return 1

    def get_time(self):
        _time = time.time() - self.time
        self.time = time.time()
        return _time

    def eval(self, best_f1, best_epoch, epoch=-1):
        # 用于验证
        self.logger.info('-' * 90)
        self.model.eval()
        all_f1, ign_f1, f1, theta = self.test_fn(self.config, self.model)
        self.model.train()
        self.logger.info(f'| epoch {epoch:3d} | time: {self.get_time():5.2f}s')
        self.logger.info('-' * 90)

        if ign_f1 > best_f1:
            best_f1 = ign_f1
            best_epoch = epoch
            if best_f1 > 0.59:
                path = opj(self.config.checkpoint_dir, f'{self.config.save_name}_{theta}.pt')
                torch.save(self.model.state_dict(), path)
                self.logger.info("Storing result...")
        return best_f1, best_epoch

    def data_trans(self, data):
        if self.config.use_gpu:
            return {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}
        else:
            return data

    def compute_output(self, output, relations, relation_mask):
        output = torch.argmax(output, dim=-1)
        output = output.data.cpu().numpy()

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if not relation_mask[i][j]:
                    break
                self.acc_total.add(relations[i][j][output[i][j]])
                if relations[i][j][0]:
                    self.acc_NA.add(output[i][j] == 0)
                else:
                    self.acc_not_NA.add(relations[i][j][output[i][j]])

    def train_log(self, epoch, global_step, total_loss):
        last_lrs = ['{:.2e}'.format(last_lr) for last_lr in self.scheduler.get_last_lr()]
        cur_loss = total_loss / self.config.log_step
        elapsed = int(self.get_time() * 1000 / self.config.log_step)
        self.logger.info(f'epoch {epoch:2d} | step {global_step:6d} | ms/b {elapsed:5d} | loss {cur_loss:.6f} '
                         f'| NA acc: {self.acc_NA.get():5.4f} | not NA acc: {self.acc_not_NA.get():5.4f} '
                         f'| tot acc: {self.acc_total.get():4.3f} | ' + ', '.join(last_lrs))

    def train(self):
        global_step, total_loss, best_f1, best_epoch = 0, 0, 0, 0
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.config.epochs):
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            for data in self.train_dataloader:
                batch = self.data_trans(data)
                relations = batch['relations']
                relation_mask = batch['relation_mask']

                if torch.sum(relation_mask) == 0:
                    # 3053篇文章中有26篇文章的labels为空
                    continue
                with autocast():
                    pred = self.model(**batch)
                    pred, loss = self.loss_fn(pred, batch)

                # loss.backward()
                scaler.scale(loss).backward()
                if self.config.clip_grad:
                    clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                scaler.step(self.optimizer)
                scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.compute_output(pred, relations, relation_mask)
                global_step += 1
                total_loss += loss.item()
                if global_step % self.config.log_step == 0:
                    self.train_log(epoch, global_step, total_loss)
                    total_loss = 0

            # if self.acc_not_NA.get() > 0.5 and
            if epoch % self.config.evaluate_epoch == 0:
                best_f1, best_epoch = self.eval(best_f1, best_epoch, epoch)
                self.get_time()

        best_f1, best_epoch = self.eval(best_f1, best_epoch)
        self.logger.info("Finish training")
        self.logger.info(f"Best epoch = {best_epoch} | F1 = {best_f1}")


if __name__ == "__main__":
    opt = option()
    con = Config(opt)

    PTM = BertModel.from_pretrained(f'bert-base-{con.data_type}')
    MODEL = models.my_model
    Model = MODEL(config=con, PTM=PTM).cuda()
    logger = get_logger(opj(con.log_dir, con.save_name))
    log_config_information(con, logger)

    model_name = f'Model name: {Model.__class__.__name__}'
    logger.info(model_name)

    # param = torch.load(opj(con.checkpoint_dir, '.pt'))
    # Model.load_state_dict(param)

    Trainer = trainer(config=con, model=Model, Logger=logger)
    Trainer.train()
