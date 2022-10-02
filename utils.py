import argparse
import json
import logging
import os
import time
from os.path import join as path_join

import numpy as np
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only

from Config import Config


def get_logger(save_name, is_print=True):
    logger = logging.getLogger('my_log')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(save_name + '.txt')
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


def log_config_information(config: Config, logger):
    """
    记录训练的参数信息
    """

    TIME = time.strftime("%m-%d %H:%M")
    logger.info(TIME)
    logger.info('=' * 25 + 'config information' + '=' * 25)
    s = []
    i = 1
    for k, v in config.__dict__.items():
        if 'dir' in k or 'path' in k:
            continue
        s.append(f'{k}: {v}')
        if i % 5 == 0:
            s.append('\n')
        i += 1
    MAX_LEN = max(map(len, s))
    message = [m if m == '\n' else m + (MAX_LEN - len(m)) * ' ' for m in s]
    message = '|'.join(message)
    logger.info('|' + message)


def get_params(model):
    """
    统计模型的可训练参数总数
    :param model: 待统计参数模型
    """
    s = 'total parameters:  ' + str(sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))
    return s


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class all_accuracy(object):
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
