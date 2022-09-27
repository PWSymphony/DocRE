
import numpy as np
import torch
import zipfile
import os
import _pickle as pickle
from torch.utils.data import Dataset, DataLoader
from itertools import permutations
from torch.nn.utils.rnn import pad_sequence


class my_dataset(Dataset):
    def __init__(self, path, is_zip=True):
        if is_zip:
            with zipfile.ZipFile(path + '.zip') as zipFile:
                file_name = os.path.split(path)[-1]
                zip_data = zipFile.read(file_name + '.data')
                self.data = pickle.loads(zip_data)
        else:
            with open(path + '.data', 'rb') as f:
                self.data = pickle.loads(f.read())

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_batch(batch):
    max_len = max(len(b['input_id']) for b in batch)
    batch_size = len(batch)
    ner_len = max(len(b['ner_id']) for b in batch)
    # max_ht_num = max(len(b['hts']) for b in batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    input_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    mention_map = []
    entity_map = []
    # entity_pos = torch.zeros(batch_size, max_len, dtype=torch.long)
    # entity_ner = torch.zeros(batch_size, ner_len, dtype=torch.long)
    # ht_pair_dis = torch.zeros(batch_size, max_ht_num, dtype=torch.long)
    hts = []

    relations = []
    relation_mask = []

    if 'graph' in batch[0]:
        graphs = [b['graph'] for b in batch]
        inter_index = [b['inter_index'] for b in batch]
        intra_index = [b['intra_index'] for b in batch]
    else:
        graphs = None
        inter_index = None
        intra_index = None

    indexes = []
    titles = []
    all_label2in_train = []
    all_test_idxs = []

    for idx, b in enumerate(batch):
        input_ids[idx, :len(b['input_id'])] = b['input_id']
        input_mask[idx, :len(b['input_id'])] = 1
        mention_map.append(b['mention_map'])
        entity_map.append(b['entity_map'])
        # entity_pos[idx, :len(b['entity_pos'])] = b['entity_pos']
        # entity_ner[idx, :len(b['ner_id'])] = b['ner_id']
        # ht_pair_dis[idx, :len(b['ht_distance'])] = b['ht_distance']
        temp_ht = np.asarray(list(permutations(range(b['entity_num']), 2)))
        hts.append(temp_ht.T)

        relations.append(b['relations'].float())
        relation_mask.append(torch.ones(b['relations'].shape[0], dtype=torch.bool))

        # for test
        titles.append(b['title'])
        indexes.append(b['index'])
        all_test_idxs.append(temp_ht.tolist())
        label2in_train = {}
        labels = b['labels']
        for label in labels:
            label2in_train[(label['h'], label['t'], label['r'])] = label['in_train']
        all_label2in_train.append(label2in_train)

    res = dict(input_id=input_ids,
               input_mask=input_mask,
               mention_map=mention_map,
               entity_map=entity_map,
               # entity_ner=entity_ner,
               # entity_pos= entity_pos,
               # ht_pair_dis=ht_pair_dis,
               hts=hts,
               relations=pad_sequence(relations, batch_first=True),
               relation_mask=pad_sequence(relation_mask, batch_first=True),

               # test
               titles=titles,
               indexes=indexes,
               labels=all_label2in_train,
               all_test_idxs=all_test_idxs)

    if graphs:
        res['graphs'] = graphs
        res['inter_index'] = inter_index
        res['intra_index'] = intra_index

    return res


if __name__ == "__main__":
    data = my_dataset(r'data/dev_uncased', is_zip=True)
    dataloader = DataLoader(data, batch_size=2, collate_fn=get_batch)
    for d in dataloader:
        pass