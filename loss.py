import json
from os.path import join as path_join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models import MAX


class ATLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.id2rel = json.load(open(path_join(config.meta_path, 'id2rel.json')))
        self.total_recall = 0
        self.test_result = []

    @staticmethod
    def forward(pred, batch):
        relations = batch['relations']
        relation_mask = batch['relation_mask']

        have_relation_num = relation_mask.sum(-1)
        new_pred = [pred[i, :index] for i, index in enumerate(have_relation_num)]
        labels = [relations[i, :index] for i, index in enumerate(have_relation_num)]
        new_pred = torch.cat(new_pred, dim=0)
        labels = torch.cat(labels, dim=0)

        # --------- 尝试加入间隔，但没效果 -----------
        # with torch.no_grad():
        #     # TH label
        #     th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        #     th_label[:, 0] = 1.0
        #     labels[:, 0] = 0.0
        #
        #     p_mask = labels + th_label  # [1, 0, 0, 1, 1, 0, 0, ...]
        #     n_mask = 1 - labels  # [1, 0,]
        #
        #     p_c = 1
        #     p_mask = p_mask.bool() & ((new_pred - new_pred[..., :1]) <= p_c)
        #     p_mask = p_mask.float()
        # # Rank positive classes to TH
        # logit1 = new_pred - (1 - p_mask) * MAX
        # loss_mask = labels.bool() & ((new_pred - new_pred[..., :1]) <= p_c)
        # loss_mask = loss_mask.float()
        # loss1 = -(F.log_softmax(logit1, dim=-1) * loss_mask).sum(-1)
        #
        # # Rank TH to negative classes
        # with torch.no_grad():
        #     n_c = 5
        #     n_mask = n_mask.bool() & (-(new_pred - new_pred[..., :1]) <= n_c)
        #     n_mask = n_mask.float()
        #
        # logit2 = new_pred - (1 - n_mask) * MAX
        # n_loss_mask = th_label.bool() & (-(new_pred - new_pred[..., :1]) <= n_c)
        # n_loss_mask = n_loss_mask.float()
        # loss2 = -(F.log_softmax(logit2, dim=-1) * n_loss_mask).sum(-1)

        # ---------- 原始AT loss -----------
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = new_pred - (1 - p_mask) * MAX
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = new_pred - (1 - n_mask) * MAX
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    @staticmethod
    def get_label(logits, label_mask, num_labels=-1):  # num_labels 是最大的标签数量
        have_relation_num = label_mask.sum(-1).cpu().detach().tolist()
        logits = [logits[i, :index] for i, index in enumerate(have_relation_num)]
        logits = torch.cat(logits, dim=0)

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        output = torch.split(output, have_relation_num, dim=0)
        output = pad_sequence(output, batch_first=True)
        return output

    @staticmethod
    def label2list(pre_label):
        pre_label = pre_label.data.cpu().numpy()
        output = []
        for b in pre_label:
            b_output = []
            for ht in b:
                b_output.append((ht.nonzero()[0]).tolist())
            output.append(b_output)
        return output

    def push_result(self, batch_result, batch_info):
        labels = batch_info['labels']
        all_test_idxs = batch_info['all_test_idxs']
        titles = batch_info['titles']
        indexes = batch_info['indexes']
        relation_mask = batch_info['relation_mask']

        pre_label = self.get_label(batch_result, relation_mask, 4)
        pre_label = self.label2list(pre_label)
        predict_re = batch_result.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            index = indexes[i]

            self.total_recall += len(label)

            test_idxs = all_test_idxs[i]
            for j, (h_idx, t_idx) in enumerate(test_idxs):
                for r in pre_label[i][j]:
                    if r == 0:
                        break
                    in_train = False
                    in_label = False
                    if (h_idx, t_idx, r) in label:
                        in_label = True
                        in_train = label[(h_idx, t_idx, r)]
                    self.test_result.append((in_label, float(predict_re[i, j, r]), in_train,
                                             titles[i], self.id2rel[str(r)], index, h_idx, t_idx, r))

    def get_result(self, is_test=False):
        if is_test:
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                      in self.test_result]
            with open(path_join(self.config.result_dir, self.config.save_name + "_result.json"), 'w') as f:
                json.dump(output, f)
            return

        correct = sum([i[0] for i in self.test_result])
        pr_x = float(correct) / self.total_recall
        pr_y = float(correct) / len(self.test_result) if len(self.test_result) else 0
        f1 = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        all_f1 = f1

        correct_in_train = sum([i[2] for i in self.test_result])
        pr_x = float(correct) / self.total_recall
        pr_y = float(correct - correct_in_train) / (len(self.test_result) + 1 - correct_in_train)

        ign_f1 = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        self.test_result = []
        self.total_recall = 0
        return dict(all_f1=round(all_f1 * 100, 2),
                    ign_f1=round(ign_f1 * 100, 2))


class BCELoss(nn.Module):
    def __init__(self, config):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.config = config
        self.id2rel = json.load(open(path_join(config.meta_path, 'id2rel.json')))
        self.total_recall = 0
        self.test_result = []

    def forward(self, pred, batch):
        relations = batch['relations']
        relation_mask = batch['relation_mask']

        relation_num = pred.shape[-1]
        pred_loss = self.loss_fn(pred, relations) * relation_mask.unsqueeze(2)
        loss = torch.sum(pred_loss) / (relation_num * torch.sum(relation_mask))

        return pred, loss

    def push_result(self, batch_result, batch_info):
        labels = batch_info['labels']  # [{(h,t,r): bool, ...}, ...]
        all_test_idxs = batch_info['all_test_idxs']  # [[(h,t), (h, t), ...], ....]
        titles = batch_info['titles']
        indexes = batch_info['indexes']

        predict_re = torch.sigmoid(batch_result)
        predict_re = predict_re.data.cpu().numpy()
        for i in range(len(labels)):
            label = labels[i]
            index = indexes[i]
            self.total_recall += len(label)
            test_idxs = all_test_idxs[i]
            for j, (h_idx, t_idx) in enumerate(test_idxs):
                for r in range(1, self.config.relation_num):
                    in_train = False
                    in_label = False
                    if (h_idx, t_idx, r) in label:
                        in_label = True
                        in_train = label[(h_idx, t_idx, r)]
                    self.test_result.append((in_label, float(predict_re[i, j, r]), in_train,
                                             titles[i], self.id2rel[str(r)], index, h_idx, t_idx, r))

    def get_result(self, input_theta=-1):
        # input_theta = -1 表明是验证集，
        self.test_result.sort(key=lambda x: x[1], reverse=True)  # 将预测结果按照降序排列

        if input_theta != -1:  # 对于测试集，不需要计算f1，只要输出结果文件
            assert 0 <= input_theta <= 1, "input_theta 需要位于 [0, 1] 之间, 或者等于-1"
            w = 0
            for i, item in enumerate(self.test_result):
                if item[1] > input_theta:
                    w = i
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                      in self.test_result[:w + 1]]
            with open(path_join(self.config.result_dir, self.config.save_name + "_result.json"), 'w') as f:
                json.dump(output, f)
            return

        pr_x = []
        pr_y = []
        correct = 0
        self.total_recall = 1 if not self.total_recall else self.total_recall

        for i, item in enumerate(self.test_result):
            correct += item[0]
            pr_x.append(float(correct) / self.total_recall)
            pr_y.append(float(correct) / (i + 1))

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        all_f1 = f1  # 没剔除训练集中出现过的实体对的f1
        theta = self.test_result[f1_pos][1]

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        ign_pos = 0
        for i, item in enumerate(self.test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / self.total_recall)
            if item[1] > theta:
                ign_pos = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        self.test_result = []
        self.total_recall = 0
        return dict(all_f1=round(all_f1 * 100, 2),
                    ign_f1=round(ign_f1 * 100, 2),
                    theta=theta,
                    ign_theta_f1=round(f1_arr[ign_pos] * 100, 2))
