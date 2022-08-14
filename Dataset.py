import torch
import zipfile
import os
import _pickle as pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


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

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_batch(batch):
    max_len = max(len(b['input_id']) for b in batch)
    batch_size = len(batch)
    ner_len = max(len(b['ner_id']) for b in batch)
    max_ht_num = max(len(b['hts']) for b in batch)
    relation_num = batch[0]['relations'].shape[-1]
    max_mention_num = max(b['mention_num'] for b in batch)
    max_entity_num = max(b['entity_num'] for b in batch)
    # mention_ht_num = max(b['mention_ht'].shape[0] for b in batch)
    # mention_pair_num = max(b['pairs'].shape[0] for b in batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    input_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    mention_map = torch.zeros(batch_size, max_mention_num, max_len, dtype=torch.float32)
    entity_map = torch.zeros(batch_size, max_entity_num, max_mention_num, dtype=torch.float)
    # entity_pos = torch.zeros(batch_size, max_len, dtype=torch.long)
    entity_ner = torch.zeros(batch_size, ner_len, dtype=torch.long)
    ht_pair_dis = torch.zeros(batch_size, max_ht_num, dtype=torch.long)
    hts = torch.zeros(batch_size, max_ht_num, 2, dtype=torch.int64)

    # mention_ht = torch.zeros(batch_size, mention_ht_num, 2, dtype=torch.int64)
    # mention_ht_map = torch.zeros(batch_size, max_ht_num, mention_ht_num, dtype=torch.float32)
    # mention_pair = torch.zeros(batch_size, mention_pair_num, 2, dtype=torch.int64)
    # mention_pair_label = torch.zeros(batch_size, mention_pair_num, 2, dtype=torch.float32)
    # mention_pair_mask = torch.zeros(batch_size, mention_pair_num, dtype=torch.bool)

    relations = torch.zeros(batch_size, max_ht_num, relation_num, dtype=torch.float32)
    relation_mask = torch.zeros(batch_size, max_ht_num, dtype=torch.bool)

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
        mention_map[idx, :b['mention_map'].shape[0], :b['mention_map'].shape[1]] = b['mention_map']
        entity_map[idx, :b['entity_map'].shape[0], :b['entity_map'].shape[1]] = b['entity_map']
        # entity_pos[idx, :len(b['entity_pos'])] = b['entity_pos']
        entity_ner[idx, :len(b['ner_id'])] = b['ner_id']
        ht_pair_dis[idx, :len(b['ht_distance'])] = b['ht_distance']
        hts[idx, :len(b['hts'])] = b['hts']

        # mention_ht[idx, :b['mention_ht'].shape[0]] = b['mention_ht']
        # mention_ht_map[idx, :b['mention_ht_map'].shape[0], :b['mention_ht_map'].shape[1]] = b['mention_ht_map']
        # mention_pair[idx, :b['pairs'].shape[0]] = b['pairs']
        # mention_pair_label[idx, :b['pair_labels'].shape[0]] = b['pair_labels']
        # mention_pair_mask[idx, :b['pair_labels'].shape[0]] = 1

        relations[idx, :len(b['relations'])] = b['relations']
        relation_mask[idx, :len(b['relations'])] = 1

        # for test
        titles.append(b['title'])
        indexes.append(b['index'])
        all_test_idxs.append(b['hts'].tolist())
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
               # mention_ht=mention_ht,
               # mention_ht_map=mention_ht_map,
               # mention_pair=mention_pair,
               # mention_pair_label=mention_pair_label,
               # mention_pair_mask=mention_pair_mask,
               relations=relations,
               relation_mask=relation_mask,

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
    data = my_dataset(r'E:\pythonProject\MYMODEL2\data\dev_uncased', is_zip=False)
    dataloader = DataLoader(data, batch_size=2, collate_fn=get_batch)
    for d in dataloader:
        pass
