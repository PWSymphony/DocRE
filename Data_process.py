import _pickle as pickle
import zipfile
from collections import defaultdict
from os.path import join as path_join

import torch
import ujson as json
from tqdm import tqdm
from transformers import BertTokenizer

raw_data_path = r"./data/raw"
out_data_path = r"./data"
train_annotated_file_name = path_join(raw_data_path, 'train_annotated.json')
dev_file_name = path_join(raw_data_path, 'dev.json')
test_file_name = path_join(raw_data_path, 'test.json')

train_revised_file_path = path_join(raw_data_path, 'train_revised.json')
dev_revised_file_path = path_join(raw_data_path, 'dev_revised.json')

with open(path_join(raw_data_path, 'rel2id.json')) as f:
    rel2id = json.load(f)

with open(path_join(raw_data_path, 'ner2id.json')) as f:
    ner2id = json.load(f)

# 在train中出现过的实体对
fact_in_train = set([])

data_type = r'uncased'
tokenizer = BertTokenizer.from_pretrained(f'bert-base-{data_type}')
token_start_id = tokenizer.cls_token_id
token_end_id = tokenizer.sep_token_id
type_relation = defaultdict(set)


def process(data_path, suffix=''):
    """
    :param data_path: 原始数据
    :param suffix: 数据类别（测试集、训练集、验证集）
    """
    with open(data_path, 'r') as file:
        ori_data = json.load(file)

    data = []

    for doc_id, doc in tqdm(enumerate(ori_data), total=len(ori_data), desc=suffix, unit='doc'):
        sent_len = [len(sent) for sent in doc['sents']]
        sent_map = [0] + [sum(sent_len[:i + 1]) for i in range(len(sent_len))]
        mention_start = set()
        mention_end = set()
        mention_num = 0
        for entity in doc['vertexSet']:
            mention_num += (len(entity))
            for mention in entity:
                mention_start.add((mention['sent_id'], mention['pos'][0]))
                mention_end.add((mention['sent_id'], mention['pos'][1] - 1))

                mention['global_pos'] = [mention['pos'][0] + sent_map[mention['sent_id']],
                                         mention['pos'][1] + sent_map[mention['sent_id']]]

        input_id = [token_start_id]
        word_map = []
        for sent_id, sent in enumerate(doc['sents']):
            for word_id, word in enumerate(sent):
                word_map.append(len(input_id))

                token = tokenizer.tokenize(word)
                if (sent_id, word_id) in mention_start:
                    token = ['*'] + token
                if (sent_id, word_id) in mention_end:
                    token = token + ['*']
                token = tokenizer.convert_tokens_to_ids(token)

                input_id.extend(token)

        assert len(word_map) == sum(sent_len)
        input_id.append(token_end_id)
        word_map.append(len(input_id))  # 加上了最后的 [102]

        mention_map = torch.zeros([mention_num, len(input_id)])
        entity_map = torch.zeros((len(doc['vertexSet']), mention_num))
        mention_id = 0
        start = 0
        for idx, entity in enumerate(doc['vertexSet']):
            entity_map[idx, start: start + len(entity)] = 1
            start += len(entity)
            for mention in entity:
                mention_map[mention_id, word_map[mention['global_pos'][0]]] = 1
                mention_id += 1

        item = {'index': doc_id,
                'title': doc['title'],
                'input_id': torch.tensor(input_id),
                'mention_num': mention_num,
                'entity_num': len(doc['vertexSet']),
                'mention_map': mention_map,
                'entity_map': entity_map}

        new_labels = []
        labels = doc.get('labels', [])  # labels: [{'r': 'P159', 'h': 0, 't': 2, 'evidence': [0]}, ...]
        train_triple = set()  # 存储训练集中的实体对 {(12, 4), (2, 4), ... }
        idx2label = defaultdict(list)
        for label in labels:
            rel = label['r']
            assert (rel in rel2id)
            label['r'] = rel2id[rel]  # 将关系代号转为关系ID, 如把 "p159" 转化为 "1"
            idx2label[(label['h'], label['t'])].append(label['r'])

            train_triple.add((label['h'], label['t']))

            if suffix == 'train':
                for e_h in doc['vertexSet'][label['h']]:
                    for e_t in doc['vertexSet'][label['t']]:
                        fact_in_train.add((e_h['name'], e_t['name'], rel))
                        label['in_train'] = True  # 占位
            else:
                label['in_train'] = False
                label['in_distant_train'] = False
                for e_h in doc['vertexSet'][label['h']]:
                    for e_t in doc['vertexSet'][label['t']]:
                        if (e_h['name'], e_t['name'], rel) in fact_in_train:
                            label['in_train'] = True

            new_labels.append(label)

        item['labels'] = new_labels

        hts = []
        relations = []
        relation_num = len(rel2id)
        for j in range(len(doc['vertexSet'])):
            for k in range(len(doc['vertexSet'])):
                if j == k:
                    continue
                relation = [0] * relation_num
                hts.append([j, k])

                if (j, k) in idx2label:
                    for rel_id in idx2label[(j, k)]:
                        relation[rel_id] = 1
                else:
                    relation[0] = 1
                relations.append(relation)

        label2in_train = {}
        for label in new_labels:
            label2in_train[(label['h'], label['t'], label['r'])] = label['in_train']

        item['hts'] = torch.tensor(hts)
        item['relations'] = torch.tensor(relations)
        item['label2in_train'] = label2in_train

        data.append(item)

    print(suffix, ': ', len(data))
    out_dir = path_join(out_data_path, f'{suffix}.zip')
    data = pickle.dumps(data)
    zip_file = zipfile.ZipFile(out_dir, mode='w', compression=zipfile.ZIP_LZMA)
    zip_file.writestr(f'{suffix}.data', data)
    print(suffix, f': {len(data) / 1024 / 1024 :.2f}MB')


def get_type_relation(file_name):
    with open(file_name, 'r') as file:
        docs = json.load(file)

    global type_relation
    for item in docs:
        entity = item['vertexSet']
        for label in item['labels']:
            h_type = entity[label['h']][0]['type']
            h_type = ner2id[h_type]

            t_type = entity[label['t']][0]['type']
            t_type = ner2id[t_type]

            r = rel2id[label['r']]

            type_relation[(h_type, t_type)].add(r)
    type_relation = {k: list(v) + [0] for k, v in type_relation.items()}


if __name__ == '__main__':
    process(train_annotated_file_name, suffix='train')
    process(dev_file_name, suffix='dev')
    process(test_file_name, suffix='test')
