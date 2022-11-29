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
        lr = metrics.pop('lr', None)
        loss = metrics.pop('loss', None)
        info = [f'{k}: {v: .2f}' if isinstance(v, float) else f'{k}: {v}' for k, v in metrics.items()]
        info = ' | '.join(info)
        pre = f'epoch: {epoch + 1: 3d} | step: {step + 1: 6d} | '

        if loss:
            pre = pre + f'loss: {loss: 5f} | '
        info = pre + info

        if lr:
            info = info + f' | lr: {lr: .3e}'
        self.base_log.info(info)

        if (step + 1) % 100 != 0:
            self.base_log.info('-' * 64)
