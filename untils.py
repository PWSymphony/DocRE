import os
import time
import numpy as np
from Config import Config
import logging


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
