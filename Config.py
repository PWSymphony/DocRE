import argparse
import os


class Config:
    def __init__(self, arg):

        self.ner_num = 7
        self.relation_num = 97
        self.dis_size = 22
        self.entity_type_size = 20
        self.coref_size = 50
        self.other_emb_size = 20
        self.loss_fn = arg.loss_fn
        self.seed = 8

        self.evaluate_epoch = 5
        self.log_step = 300

        self.data_path = r"./data"
        self.data_type = r'uncased'
        self.train_prefix = 'train'
        if arg.is_test:
            self.test_prefix = 'test'
        else:
            self.test_prefix = 'dev'

        self.checkpoint_dir = r'./checkpoint'
        self.result_dir = r'./result'
        self.log_dir = r'./log'
        if not os.path.exists("result"):
            os.mkdir("result")
        if not os.path.exists("log"):
            os.mkdir("log")
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        self.hidden_size = arg.hidden_size
        self.dropout = arg.dropout
        self.clip_grad = True

        self.use_gpu = not arg.cpu
        self.epochs = arg.epoch
        self.batch_size = arg.batch_size
        self.lr = arg.lr
        self.warm_ratio = 0
        self.decay_ratio = 0
        self.theta = arg.theta
        self.pre_lr = arg.pre_lr

        self.save_name = arg.save_name


def option():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--is_test', action='store_true', default=False)

    # train
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--pre_lr', type=float, default=5e-5)
    parser.add_argument('--loss_fn', type=str, default='ATL')

    # test
    parser.add_argument('--theta', type=float, default=-1)

    # save
    parser.add_argument('--save_name', type=str, default='test_code')

    return parser.parse_args()


if __name__ == '__main__':
    o = option()
    c = Config(o)

    s = '\n===config information===\n'
    for k, v in c.__dict__.items():
        if 'dir' in k or 'path' in k:
            continue
        s += f'{k}: {v}\n'
    print(s)
