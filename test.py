import json
from os.path import join as opj
import time
import models
import numpy as np
import torch
# from sklearn import metrics
from torch.utils.data import DataLoader
from Config import Config, option
from Dataset import my_dataset, get_batch
from untils import my_logging
from transformers import BertModel
from transformers import logging as log
from loss import ATLoss

log.set_verbosity_error()


def test(config: Config, model, output=False, input_theta=-1.0):
    id2rel = json.load(open(opj(config.data_path, 'raw', 'id2rel.json')))
    test_result = []
    total_recall = 0  # 有关系的实体对数量(有多少种关系，就记为多少对实体)
    predicted_as_zero = 0
    total_ins_num = 0  # 总共预测的实体对数量
    top_acc = 0
    hava_label = 0  # 有关系的实体对数量(只要有关系就算1对，不重复记录)

    test_data = my_dataset(opj(config.data_path, f'{config.test_prefix}_{config.data_type}'))

    dataloader = DataLoader(test_data,
                            batch_size=config.batch_size,
                            collate_fn=get_batch)

    for data_idx, data in enumerate(dataloader):  # data_idx用于控制输出信息
        with torch.no_grad():
            batch = {}
            if config.use_gpu:
                for k, v in data.items():
                    if torch.is_tensor(v):
                        v = v.cuda()
                    batch[k] = v
            else:
                batch = data

            labels = batch['labels']  # [{(h,t,r): bool, ...}, ...]
            all_test_idxs = batch['all_test_idxs']  # [[(h,t), (h, t), ...], ....]
            titles = batch['titles']
            indexes = batch['indexes']

            predict_re = model(**batch)
            if config.loss_fn == 'MY':
                predict_re = predict_re[0]
            predict_re = torch.sigmoid(predict_re)
            predict_re = predict_re.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            index = indexes[i]

            total_recall += len(label)

            test_idxs = all_test_idxs[i]
            for j, (h_idx, t_idx) in enumerate(test_idxs):
                r = np.argmax(predict_re[i, j])
                predicted_as_zero += (r == 0)
                total_ins_num += 1

                if (h_idx, t_idx, r) in label:
                    top_acc += 1

                flag = False
                for r in range(1, config.relation_num):
                    in_train = False
                    in_label = False
                    if (h_idx, t_idx, r) in label:
                        flag = True
                        in_label = True
                        in_train = label[(h_idx, t_idx, r)]
                    test_result.append((in_label, float(predict_re[i, j, r]), in_train,
                                        titles[i], id2rel[str(r)], index, h_idx, t_idx, r))

                if flag:
                    hava_label += 1

    test_result.sort(key=lambda x: x[1], reverse=True)

    print('total_recall     ', total_recall)
    print('predicted as zero', predicted_as_zero)
    print('total ins num    ', total_ins_num)
    print('top_acc          ', top_acc)

    pr_x = []
    pr_y = []
    correct = 0
    w = 0

    if total_recall == 0:
        total_recall = 1  # for test

    for i, item in enumerate(test_result):
        correct += item[0]
        pr_x.append(float(correct) / total_recall)
        pr_y.append(float(correct) / (i + 1))
        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    all_f1 = f1  # 没剔除训练集中出现过的实体对的f1
    theta = test_result[f1_pos][1]

    if input_theta == -1:
        w = f1_pos
        input_theta = theta

    if config.test_prefix == 'dev':
        my_logging('Dev(All) | theta {:3.4f} | f1 {:3.4f}'.format(theta, f1), config.save_name)
    if config.test_prefix == 'test':
        my_logging('Test(All) | max_f1 {:3.4f} | input_theta {:3.4f} | test_result f1 {:3.4f}'.
                   format(f1, input_theta, f1_arr[w]), config.save_name)

    if output:
        output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                  in test_result[:w + 1]]
        with open(opj(config.result_dir, config.save_name + "_" + config.test_prefix + "_result.json"), 'w') as f:
            json.dump(output, f)
        print('finish output')
        return

    pr_x = []
    pr_y = []
    correct = correct_in_train = 0
    w = 0
    for i, item in enumerate(test_result):
        correct += item[0]
        if item[0] & item[2]:
            correct_in_train += 1
        if correct_in_train == correct:
            p = 0
        else:
            p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
        pr_y.append(p)
        pr_x.append(float(correct) / total_recall)
        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    ign_f1 = f1_arr.max()

    my_logging('Ignore max_f1 {:3.4f} | input_theta {:3.4f} | test_result f1 {:3.4f}'.
               format(ign_f1, input_theta, f1_arr[w]), config.save_name)

    return all_f1, ign_f1, f1_arr[w], input_theta


def ATLtest(config: Config, model, output=False):
    id2rel = json.load(open(opj(config.data_path, 'raw', 'id2rel.json')))
    loss_fn = ATLoss()

    test_result = []
    total_recall = 0  # 有关系的实体对数量(有多少种关系，就记为多少对实体)
    predicted_as_zero = 0
    total_ins_num = 0  # 总共预测的实体对数量
    top_acc = 0
    hava_label = 0  # 有关系的实体对数量(只要有关系就算1对，不重复记录)

    test_data = my_dataset(opj(config.data_path, f'{config.test_prefix}_{config.data_type}' + '.pkl'))

    dataloader = DataLoader(test_data,
                            batch_size=config.batch_size,
                            collate_fn=get_batch)

    for data_idx, data in enumerate(dataloader):  # data_idx用于控制输出信息
        with torch.no_grad():
            batch = {}
            if config.use_gpu:
                for k, v in data.items():
                    if torch.is_tensor(v):
                        v = v.cuda()
                    batch[k] = v
            else:
                batch = data

            labels = batch['labels']  # [{(h,t,r): bool, ...}, ...]
            all_test_idxs = batch['all_test_idxs']  # [[(h,t), (h, t), ...], ....]
            titles = batch['titles']
            indexes = batch['indexes']

            relation_mask = batch['relation_mask']

            predict_re = model(**batch)
            pre_label = loss_fn.get_label(predict_re, relation_mask, 4)
            pre_label = label2list(pre_label)
            predict_re = predict_re.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            index = indexes[i]

            total_recall += len(label)

            test_idxs = all_test_idxs[i]
            for j, (h_idx, t_idx) in enumerate(test_idxs):
                r = np.argmax(predict_re[i, j])
                predicted_as_zero += (r == 0)
                total_ins_num += 1

                if (h_idx, t_idx, r) in label:
                    top_acc += 1

                flag = False
                for r in pre_label[i][j]:
                    if r == 0:
                        break
                    in_train = False
                    in_label = False
                    if (h_idx, t_idx, r) in label:
                        flag = True
                        in_label = True
                        in_train = label[(h_idx, t_idx, r)]
                    test_result.append((in_label, float(predict_re[i, j, r]), in_train,
                                        titles[i], id2rel[str(r)], index, h_idx, t_idx, r))

                if flag:
                    hava_label += 1

    print('total_recall     ', total_recall)
    print('predicted as zero', predicted_as_zero)
    print('total ins num    ', total_ins_num)
    print('top_acc          ', top_acc)

    if total_recall == 0:
        total_recall = 1  # for test

    correct = sum([i[0] for i in test_result])
    pr_x = float(correct) / total_recall
    pr_y = float(correct) / len(test_result)
    f1 = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    all_f1 = f1

    if config.test_prefix == 'dev':
        my_logging('Dev(All) | f1 {:3.4f}'.format(f1), config.save_name)
    if config.test_prefix == 'test':
        my_logging('Test(All) | test_result f1 {:3.4f}'.
                   format(f1), config.save_name)

    if output:
        output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                  in test_result]
        with open(opj(config.result_dir, config.save_name + "_" + config.test_prefix + "_result.json"), 'w') as f:
            json.dump(output, f)
        print('finish output')
        return

    correct_in_train = sum([i[2] for i in test_result])
    pr_x = float(correct) / total_recall
    pr_y = float(correct - correct_in_train) / (len(test_result) + 1 - correct_in_train)

    ign_f1 = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))

    my_logging('Ignore f1 {:3.4f}'.format(ign_f1), config.save_name)

    return all_f1, ign_f1, -1, -1


def label2list(pre_label):
    pre_label = pre_label.data.cpu().numpy()
    output = []
    for b in pre_label:
        b_output = []
        for ht in b:
            b_output.append((ht.nonzero()[0]).tolist())
        output.append(b_output)
    return output


def test_all(config: Config):
    # 在测试集中，theta不能为-1
    assert config.theta != -1
    # 加载模型
    MODEL = models.my_model
    PTM = BertModel.from_pretrained(r'bert-base-uncased')
    Model = MODEL(config=config, PTM=PTM).cuda()

    # 载入模型参数
    model_path = opj(config.checkpoint_dir, config.save_name + '.pt')
    s = 'load model from ' + model_path
    my_logging(s, config.save_name)
    model_data = torch.load(model_path)
    Model.load_state_dict(model_data)

    con.save_name = 'test_' + con.save_name
    test(config=config, model=Model, output=True, input_theta=config.theta)


if __name__ == "__main__":
    opt = option()
    con = Config(opt)
    test_all(con)
