import json
from itertools import permutations
from os.path import join as path_join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class ATLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.id2rel = json.load(open(path_join(config.data_path, 'raw', 'id2rel.json')))
        self.total_recall = 0
        self.test_result = []

    @staticmethod
    def forward(pred, **batch):
        labels = batch['relations']
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = pred - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = pred - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return pred, loss

    @staticmethod
    def get_label(logits, num_labels=-1):  # num_labels 是最大的标签数量
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        return output

    @staticmethod
    def label2list(pre_label):
        output = []
        for b in pre_label:
            b_output = []
            for ht in b:
                b_output.append((ht.nonzero()[0]).tolist())
            output.append(b_output)
        return output

    def push_result(self, batch_result, batch_info):
        labels = batch_info['labels']
        titles = batch_info['titles']
        indexes = batch_info['indexes']

        pre_label = self.get_label(batch_result, num_labels=4)
        pre_label = torch.split(pre_label, [x.shape[1] for x in batch_info['hts']], dim=0)
        pre_label = self.label2list(pre_label)
        for i in range(len(labels)):
            label = labels[i]
            index = indexes[i]
            self.total_recall += len(label)
            j = -1
            for h_idx, t_idx in permutations(range(batch_info['entity_map'][i].shape[0]), 2):
                j += 1
                for r in pre_label[i][j]:
                    if r == 0:
                        break
                    in_train = False
                    in_label = False
                    if (h_idx, t_idx, r) in label:
                        in_label = True
                        in_train = label[(h_idx, t_idx, r)]
                    self.test_result.append((in_label, in_train,
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
        return dict(all_f1=all_f1,
                    theta=0,
                    ign_f1=ign_f1,
                    ign_theta_f1=0)


class BCELoss(nn.Module):
    def __init__(self, config):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.config = config
        self.id2rel = json.load(open(path_join(config.data_path, 'raw', 'id2rel.json')))
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
        return dict(all_f1=all_f1,
                    theta=theta,
                    ign_f1=ign_f1,
                    ign_theta_f1=f1_arr[ign_pos])


class MultiLoss(ATLoss):
    def __init__(self, config):
        super(MultiLoss, self).__init__(config)
        self.ATLoss = ATLoss(config)
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.id2rel = json.load(open(path_join(config.data_path, 'raw', 'id2rel.json')))
        self.total_recall = 0
        self.test_result = []

    def forward(self, pred, **batch):
        if isinstance(pred, tuple):
            ent_pred, cls_pred = pred
            _, loss1 = self.ATLoss(ent_pred, **batch)
            cls_label = batch['relations'].sum(1).bool().float()
            cls_label[:, 0] = 0
            loss2 = self.BCELoss(cls_pred, cls_label)
            return ent_pred, loss1 + loss2

        else:
            pred, loss1 = self.ATLoss(pred, **batch)
            return pred, loss1


class LackLoss(ATLoss):
    def __init__(self, config):
        super(LackLoss, self).__init__(config)
        self.ATLoss = ATLoss(config)
        self.id2rel = json.load(open(path_join(config.data_path, 'raw', 'id2rel.json')))

    def forward(self, pred, **batch):
        if isinstance(pred, tuple):
            all_pred, lack_pred = pred
            all_pred, loss1 = self.ATLoss(all_pred, **batch)
            _, loss2 = self.ATLoss(lack_pred, relations=batch['lack_relations'],
                                   relation_mask=batch['lack_relation_mask'])
            return all_pred, loss1 + 0.1 * loss2

        else:
            pred, loss1 = self.ATLoss(pred, **batch)
            return pred, loss1


class CELoss(nn.Module):
    def __init__(self, config):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.gold_num = 0
        self.pred_num = 0
        self.pred_true = 0

    def forward(self, pred, **batch):
        label = batch['relations'][..., 1:].sum(-1).bool().long()
        loss = self.loss_fn(pred, label)

        with torch.no_grad():
            new_pred = pred.argmax(dim=-1).bool()
            label = label.bool()
            self.gold_num += label.sum()
            self.pred_num += new_pred.sum()
            self.pred_true += (new_pred & label).sum()

        return pred, loss.mean()

    def push_result(self, *args, **kwargs):
        pass

    def get_result(self, *args, **kwargs):
        self.pred_num = self.pred_num if self.pred_num else 1
        self.gold_num = self.gold_num if self.gold_num else 1
        recall = self.pred_true / self.gold_num
        precision = self.pred_true / self.pred_num
        f1 = (2 * recall * precision) / (recall + precision)
        return dict(all_f1=f1,
                    theta=0,
                    ign_f1=recall,
                    ign_theta_f1=precision)
