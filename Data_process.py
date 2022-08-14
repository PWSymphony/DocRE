import _pickle as pickle
import numpy as np
import torch
import ujson as json
from os.path import join as path_join
from collections import defaultdict
import dgl
from tqdm import tqdm
from transformers import BertTokenizer
import zipfile
# import spacy
from sklearn.utils import shuffle

# import networkx as nx

# spacy.prefer_gpu(0)
# nlp = spacy.load("en_core_web_sm")

raw_data_path = r"./data/raw"
out_data_path = r"./data/with_graph"
train_annotated_file_name = path_join(raw_data_path, 'train_annotated.json')
dev_file_name = path_join(raw_data_path, 'dev.json')
test_file_name = path_join(raw_data_path, 'test.json')

train_revised_file_path = path_join(raw_data_path, 'train_revised.json')
dev_revised_file_path = path_join(raw_data_path, 'dev_revised.json')

with open(path_join(raw_data_path, 'rel2id.json')) as f:
    rel2id = json.load(f)

with open(path_join(raw_data_path, 'ner2id.json')) as f:
    ner2id = json.load(f)

dis2idx = np.zeros(1024, dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9
dis2idx[512:] = 10

# 在train中出现过的实体对
fact_in_train = set([])

data_type = r'uncased'
tokenizer = BertTokenizer.from_pretrained(f'bert-base-{data_type}')
token_start_id = tokenizer.cls_token_id
token_end_id = tokenizer.sep_token_id


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
        ner_id = []
        mention_num = 0
        entity2sent = defaultdict(set)
        for entity_id, entity in enumerate(doc['vertexSet']):
            mention_num += (len(entity))
            ner_id.append(ner2id[entity[0]['type']])
            for mention in entity:
                mention_start.add((mention['sent_id'], mention['pos'][0]))
                mention_end.add((mention['sent_id'], mention['pos'][1] - 1))

                mention['global_pos'] = [mention['pos'][0] + sent_map[mention['sent_id']],
                                         mention['pos'][1] + sent_map[mention['sent_id']]]

                entity2sent[entity_id].add(mention['sent_id'])

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
        entity_pos = torch.zeros([len(input_id)])
        entity_first_appear = []
        mention_id = 0
        start = 0
        for idx, entity in enumerate(doc['vertexSet']):
            entity_map[idx, start: start + len(entity)] = 1
            start += len(entity)
            entity_first_appear.append(entity[0]['global_pos'][0])
            for mention in entity:
                mention_map[mention_id, word_map[mention['global_pos'][0]]] = 1
                entity_pos[word_map[mention['global_pos'][0]]: word_map[mention['global_pos'][1]]] = idx + 1
                mention_id += 1

        item = {'index': doc_id,
                'title': doc['title'],
                'input_id': torch.tensor(input_id),
                'ner_id': torch.tensor(ner_id),
                'mention_num': mention_num,
                'entity_num': len(doc['vertexSet']),
                'mention_map': mention_map,
                'entity_map': entity_map,
                'entity_pos': entity_pos}

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
        ht_distance = []
        g = defaultdict(list)  # 创建图
        inter_index = []
        intra_index = []
        i = -1
        for j in range(len(doc['vertexSet'])):
            for k in range(len(doc['vertexSet'])):
                if j == k:
                    continue
                i += 1
                relation = [0] * relation_num
                hts.append([j, k])
                dis = entity_first_appear[k] - entity_first_appear[j]
                if dis < 0:
                    ht_distance.append(-int(-dis2idx[dis]))
                else:
                    ht_distance.append(int(dis2idx[dis]))

                if (j, k) in idx2label:
                    for rel_id in idx2label[(j, k)]:
                        relation[rel_id] = 1
                else:
                    relation[0] = 1
                relations.append(relation)

                if entity2sent[j] & entity2sent[k]:
                    g[('entity', 'intra', 'entity')].append((j, k))  # 实体在同一个句子中出现过
                    intra_index.append(i)
                else:
                    g[('entity', 'inter', 'entity')].append((j, k))  # 实体出现在不同的句子中
                    inter_index.append(i)

        item['hts'] = torch.tensor(hts)
        item['relations'] = torch.tensor(relations)
        item['ht_distance'] = torch.tensor(ht_distance)
        item['graph'] = dgl.heterograph(g)
        item['intra_index'] = intra_index
        item['inter_index'] = inter_index

        # ht_num = len(hts)
        # mention_ht, mention_ht_map = mention_pair(doc['vertexSet'], ht_num)
        # item['mention_ht'] = torch.tensor(mention_ht)
        # item['mention_ht_map'] = mention_ht_map
        #
        # pairs, pair_labels = mention2mention(doc['vertexSet'])
        # pairs, pair_labels = shuffle(pairs, pair_labels)
        # item['pairs'] = torch.tensor(pairs)
        # item['pair_labels'] = torch.tensor(pair_labels)

        data.append(item)

    print(suffix, ': ', len(data))
    out_dir = path_join(out_data_path, f'{suffix}_{data_type}')
    data = pickle.dumps(data)
    zip_file = zipfile.ZipFile(out_dir + '.zip', mode='w', compression=zipfile.ZIP_LZMA)
    zip_file.writestr(f'{suffix}_{data_type}' + '.data', data)


def mention_pair(entities, ht_num):
    mention_num = list(map(len, entities))
    mention_ht = []
    mention_ht_num = []
    mention_index = [sum(mention_num[:i]) for i in range(len(mention_num))]
    for h_id, h in enumerate(entities):
        for t_id, t in enumerate(entities):
            if h_id == t_id:
                continue
            mention_ht_num.append(len(h) * len(t))
            for h_mention in range(len(h)):
                for t_mention in range(len(t)):
                    mention_ht.append([mention_index[h_id] + h_mention, mention_index[t_id] + t_mention])
    mention_ht_map = torch.zeros(ht_num, len(mention_ht))
    start = 0
    for index, num in enumerate(mention_ht_num):
        mention_ht_map[index, start:start + num] = 1
        start += num

    return mention_ht, mention_ht_map


def mention2mention(entities):
    mention2entity = {}
    mention_id = 0
    entity_id = 0
    pairs = []
    labels = []
    for e in entities:
        for m in e:
            mention2entity[mention_id] = entity_id
            mention_id += 1
        entity_id += 1
    for i in range(mention_id):
        for j in range(i + 1, mention_id):
            pairs.append([i, j])
            labels.append([1, 0] if mention2entity[i] == mention2entity[j] else [0, 1])

    return pairs, labels


if __name__ == '__main__':
    process(train_annotated_file_name, suffix='train')
    process(dev_file_name, suffix='dev')
    process(test_file_name, suffix='test')
