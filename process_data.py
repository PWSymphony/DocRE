import _pickle as pickle
import argparse
import json
import zipfile
from collections import defaultdict
from itertools import permutations
from os.path import join as path_join

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class Processor:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        if args.data_type == 'revised':
            self.train_file_name = path_join(args.raw_path, 'train_revised.json')
            self.dev_file_name = path_join(args.raw_path, 'dev_revised.json')
            self.test_file_name = path_join(args.raw_path, 'test_revised.json')
        else:
            self.train_file_name = path_join(args.raw_path, 'train.json')
            self.dev_file_name = path_join(args.raw_path, 'dev.json')
            self.test_file_name = path_join(args.raw_path, 'test.json')

        with open(path_join(args.meta_path, 'rel2id.json')) as f:
            self.rel2id = json.load(f)

        with open(path_join(args.meta_path, 'ner2id.json')) as f:
            self.ner2id = json.load(f)

        # 在train中出现过的实体对
        self.fact_in_train = set([])

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.type_relation = self.get_type_relation(self.train_file_name)

    def __call__(self):
        self.process(self.train_file_name, suffix='train')
        self.process(self.dev_file_name, suffix='dev')
        self.process(self.test_file_name, suffix='test')

    def process(self, data_path, suffix=''):
        with open(data_path, 'r') as file:
            ori_data = json.load(file)

        data = []

        for doc_id, doc in tqdm(enumerate(ori_data), total=len(ori_data), desc=suffix, unit='doc'):
            sent_len = [len(sent) for sent in doc['sents']]
            sent_mapping = [0] + [sum(sent_len[:i + 1]) for i in range(len(sent_len))]
            mention_start = set()
            mention_end = set()
            mention_num = 0
            for entity in doc['vertexSet']:
                mention_num += (len(entity))
                for mention in entity:
                    mention_start.add((mention['sent_id'], mention['pos'][0]))
                    mention_end.add((mention['sent_id'], mention['pos'][1] - 1))

                    mention['global_pos'] = [mention['pos'][0] + sent_mapping[mention['sent_id']],
                                             mention['pos'][1] + sent_mapping[mention['sent_id']]]

            input_id = [self.tokenizer.cls_token_id]
            word_map = []
            for sent_id, sent in enumerate(doc['sents']):
                for word_id, word in enumerate(sent):
                    word_map.append(len(input_id))

                    token = self.tokenizer.tokenize(word)
                    if (sent_id, word_id) in mention_start:
                        token = ['*'] + token
                    if (sent_id, word_id) in mention_end:
                        token = token + ['*']
                    token = self.tokenizer.convert_tokens_to_ids(token)
                    input_id.extend(token)

            assert len(word_map) == sum(sent_len)
            input_id.append(self.tokenizer.sep_token_id)
            word_map.append(len(input_id))  # 加上了最后的 sep_token_id

            sent_map = torch.zeros((len(sent_len) + 1, len(input_id)))
            sent_map[0][[0, -1]] = 1
            for i in range(len(sent_len)):
                sent_map[i + 1, word_map[sent_mapping[i]]: word_map[sent_mapping[i + 1]]] = 1

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
                    'entity_map': entity_map,
                    'sent_map': sent_map}

            new_labels = []
            labels = doc.get('labels', [])  # labels: [{'r': 'P159', 'h': 0, 't': 2, 'evidence': [0]}, ...]
            train_triple = set()  # 存储训练集中的实体对 {(12, 4), (2, 4), ... }
            idx2label = defaultdict(list)
            idx2evi = defaultdict(list)
            for label in labels:
                rel = label['r']
                assert (rel in self.rel2id)
                label['r'] = self.rel2id[rel]  # 将关系代号转为关系ID, 如把 "p159" 转化为 "1"
                idx2label[(label['h'], label['t'])].append(label['r'])
                idx2evi[(label['h'], label['t'])].extend([i + 1 for i in label['evidence']])

                train_triple.add((label['h'], label['t']))

                if suffix == 'train':
                    for e_h in doc['vertexSet'][label['h']]:
                        for e_t in doc['vertexSet'][label['t']]:
                            self.fact_in_train.add((e_h['name'], e_t['name'], rel))
                            label['in_train'] = True  # 占位
                else:
                    label['in_train'] = False
                    label['in_distant_train'] = False
                    for e_h in doc['vertexSet'][label['h']]:
                        for e_t in doc['vertexSet'][label['t']]:
                            if (e_h['name'], e_t['name'], rel) in self.fact_in_train:
                                label['in_train'] = True

                new_labels.append(label)

            relation_num = len(self.rel2id)
            hts = [list(x) for x in permutations(range(len(doc['vertexSet'])), 2)]
            relations = torch.zeros((len(hts), relation_num), dtype=torch.bool)

            evidences = torch.zeros((len(hts), len(sent_len) + 1), dtype=torch.bool)

            i = -1
            for j, k in hts:
                i += 1
                relations[i][idx2label.get((j, k), [0])] = True
                evidences[i][idx2evi.get((j, k), [0])] = True

            label2in_train = {}
            for label in new_labels:
                label2in_train[(label['h'], label['t'], label['r'])] = label['in_train']

            item['labels'] = new_labels
            item['hts'] = torch.tensor(hts)
            item['relations'] = relations
            item['label2in_train'] = label2in_train
            item['evidences'] = evidences
            data.append(item)

        print(suffix, ': ', len(data))
        out_dir = path_join(self.args.data_path, f'{suffix}.zip')
        data = pickle.dumps(data)
        zip_file = zipfile.ZipFile(out_dir, mode='w', compression=zipfile.ZIP_LZMA)
        zip_file.writestr(f'{suffix}.data', data)
        print(suffix, f': {len(data) / 1024 / 1024 :.2f}MB')

    def get_type_relation(self, file_name):
        type_relation = defaultdict(set)
        with open(file_name, 'r') as file:
            docs = json.load(file)
        for item in docs:
            entity = item['vertexSet']
            for label in item['labels']:
                h_type = entity[label['h']][0]['type']
                h_type = self.ner2id[h_type]

                t_type = entity[label['t']][0]['type']
                t_type = self.ner2id[t_type]

                r = self.rel2id[label['r']]

                type_relation[(h_type, t_type)].add(r)
        h_relation = defaultdict(set)
        t_relation = defaultdict(set)
        for k, v in type_relation.items():
            h_relation[k[0]].update(v)
            t_relation[k[1]].update(v)

        for k in type_relation.keys():
            type_relation[k].update(h_relation[k[0]])
            type_relation[k].update(t_relation[k[1]])

        return {k: list(v) + [0] for k, v in type_relation.items()}
