import argparse
import logging
import os
import time
from os.path import join as path_join

import numpy as np
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only


def get_logger(save_name, is_print=True):
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(save_name + '.txt', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if is_print:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


def my_logging(s, save_name, print_=True, log_=True, log_time=True):
    if log_time:
        s = time.strftime("%m-%d %H:%M") + ' ' + s
    if print_:
        print(s)
    if log_:
        with open(os.path.join(os.path.join("log", save_name + '.txt')), 'a+') as f_log:
            f_log.write(s + '\n')


def get_params(model):
    """
    统计模型的可训练参数总数
    :param model: 待统计参数模型
    """
    s = 'total parameters:  ' + str(sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))
    return s


class AllAccuracy(object):
    def __init__(self):
        self.NA_correct = 0
        self.NA_num = 0

        self.not_NA_correct = 0
        self.not_NA_num = 0

    def add_NA(self, num, correct_num):
        self.NA_correct += correct_num
        self.NA_num += num

    def add_not_NA(self, num, correct_num):
        self.not_NA_num += num
        self.not_NA_correct += correct_num

    def get(self):
        NA_acc = self.NA_correct / self.NA_num if self.NA_num else 0
        not_NA_acc = self.not_NA_correct / self.not_NA_num if self.not_NA_num else 0
        if self.NA_num + self.not_NA_num:
            total_acc = (self.NA_correct + self.not_NA_correct) / (self.NA_num + self.not_NA_num)
        else:
            total_acc = 0
        return dict(NA=round(float(NA_acc) * 100, 2),
                    not_NA=round(float(not_NA_acc) * 100, 2),
                    total=round(float(total_acc) * 100, 2))

    def clear(self):
        self.NA_correct = 0
        self.NA_num = 0
        self.not_NA_correct = 0
        self.not_NA_num = 0


class F1(object):
    def __init__(self):
        self.total = 0
        self.true = 0
        self.gold = 0

    def add(self, total, true, gold):
        self.total += total
        self.true += true
        self.gold += gold

    def get(self):
        precision = self.true / (self.gold + 1e-20)
        recall = self.true / (self.total + 1e-20)
        f1 = 2 * precision * recall / (recall + precision + 1e-20)
        return dict(precision=round(float(precision) * 100, 2),
                    recall=round(float(recall) * 100, 2),
                    f1=round(float(f1) * 100, 2))

    def clear(self):
        self.total = 0
        self.true = 0
        self.gold = 0


class MyLogger(LightningLoggerBase):
    def __init__(self, args):
        super(MyLogger, self).__init__()
        log_path = path_join(args.log_path, args.save_name)

        pl_logger = logging.getLogger('pytorch_lightning')
        pl_logger.addHandler(logging.FileHandler(log_path + '.txt'))

        self.base_log = get_logger(log_path, is_print=not args.hidden_log)

        cur_time = time.strftime('%Y年%m月%d日, %H:%M', time.localtime())
        self.base_log.info(cur_time)
        self.max_f1 = 0.0

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return time.strftime('%m.%d_%H:%M')

    @rank_zero_only
    def log_hyperparams(self, params):
        self.base_log.info('-' * 32 + 'config information' + '-' * 32)
        args = {}
        for k in params.keys():
            if isinstance(params[k], argparse.Namespace):
                args.update(params[k].__dict__)
            else:
                args[k] = params[k]
        i = 1
        message = []
        for k, v in args.items():
            message.append(f'{k}: {v} ')
            if i % 2 == 0:
                message.append('\n')
            i += 1
        max_len = max(map(len, message))
        message = [m if m == '\n' else m + (max_len - len(m)) * ' ' for m in message]
        self.base_log.info('|' + '|'.join(message))
        self.base_log.info('-' * 82)

    @rank_zero_only
    def log_metrics(self, metrics=None, step=None):
        epoch = int(metrics.pop('epoch'))
        if epoch == -1:
            self.base_log.info('-' * 64)
            return

        lr = metrics.pop('lr', None)
        loss = metrics.pop('loss', None)
        info = [f'{k}: {v: 3.2f}' if isinstance(v, float) else f'{k}: {v}' for k, v in metrics.items()]
        info = ' | '.join(info)
        pre = f'epoch: {epoch + 1: 3d} | step: {step + 1: 6d} | '

        if loss:
            pre = pre + f'loss: {loss: 5f} | '
        info = pre + info

        if lr:
            info = info + f' | lr: {lr: .3e}'
        self.base_log.info(info)

# 备份
# class PlModel(pl.LightningModule):
#     def __init__(self, args: argparse.Namespace):
#         super(PlModel, self).__init__()
#         self.args = args
#         self.model = Temp(args)
#         self.loss_fn = LOSS_FN[args.loss_fn](args)
#         self.loss_list = []
#         self.all_acc = AllAccuracy()
#         self.f1 = F1()
#         self.save_hyperparameters(logger=True)
#
#     def configure_optimizers(self):
#         total_step = self.args.total_step
#         PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' in n)]
#         not_PLM = [p for n, p in self.named_parameters() if p.requires_grad and ('bert' not in n)]
#         optimizer = optim.AdamW([{'params': PLM, 'lr': self.args.pre_lr},
#                                  {'params': not_PLM, 'lr': self.args.lr}])
#         scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
#                                                     num_warmup_steps=int(total_step * self.args.warm_ratio),
#                                                     num_training_steps=total_step)
#         return {"optimizer": optimizer,
#                 "lr_scheduler": {"scheduler": scheduler,
#                                  "interval": 'step'}}
#
#     def forward(self, batch):
#         return self.model(**batch)
#
#     def training_step(self, batch, batch_idx):
#         output = self.model(**batch)
#         pred, loss = self.loss_fn(pred=output, batch=batch)
#         self.compute_output(output=pred, label=batch['relations'], mask=batch['relation_mask'], compute_NA=True)
#
#         final_loss = loss  # + bin_loss
#         self.loss_list.append(final_loss)
#         log_dict = self.all_acc.get()
#         log_dict['loss'] = torch.stack(self.loss_list).mean()
#         log_dict['lr'] = self.lr_schedulers().get_last_lr()[0]
#         log_dict['epoch'] = float(self.current_epoch)
#         self.log_dict(log_dict, prog_bar=False)
#         return final_loss
#
#     def training_epoch_end(self, outputs):
#         self.all_acc.clear()
#         self.f1.clear()
#         self.loss_list = []
#
#     def validation_step(self, batch, batch_idx):
#         output = self.model(**batch)
#         self.loss_fn.push_result(output, batch)
#         pred, loss = self.loss_fn(pred=output, batch=batch)
#         return loss
#
#     def validation_epoch_end(self, validation_step_outputs):
#         dev_result = self.loss_fn.get_result()
#         dev_result.update(self.f1.get())
#         self.f1.clear()
#         self.log_dict(dev_result, prog_bar=False)
#
#     def test_step(self, batch, batch_idx):
#         output = self.model(**batch)
#         self.loss_fn.push_result(output, batch)
#
#     def test_epoch_end(self, outputs):
#         self.loss_fn.get_result(is_test=True)
#
#     def compute_output(self, output, label, mask=None, compute_NA=False):
#         with torch.no_grad():
#             cur_label = label.bool()
#             top_index = F.one_hot(torch.argmax(output, dim=-1),
#                                   num_classes=output.shape[-1]).bool()
#             result = top_index & cur_label
#             if mask is not None:
#                 if len(mask.shape) < len(cur_label.shape):
#                     cur_mask = mask.unsqueeze(2)
#                 result = result & cur_mask
#             # gold.sum(0).sum(0) 一对实体有多种关系会被计算为多对实体
#             gold_na = cur_label[..., 0].sum()
#             gold_not_na = cur_label[..., 1:].sum(-1).bool().sum()  # 先求和，在将不为0的改为1，再次求和
#             pred_na = result[..., 0].sum()
#             pred_not_na = result[..., 1:].sum()
#             self.all_acc.add_NA(num=gold_na, correct_num=pred_na)
#             self.all_acc.add_not_NA(num=gold_not_na, correct_num=pred_not_na)
