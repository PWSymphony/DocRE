import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class my_dataset(Dataset):
    def __init__(self, path, file_name):
        file_path = os.path.join(path, file_name)
        self.data = torch.load(file_path + '.pt')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_batch(batch):
    max_len = max([len(b['input_id']) for b in batch])
    batch_size = len(batch)
    max_ht_num = max([len(b['hts']) for b in batch])
    relation_num = batch[0]['relations'].shape[-1]
    max_mention_num = max([b['mention_num'] for b in batch])
    max_entity_num = max([b['entity_num'] for b in batch])

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    input_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    mention_map = torch.zeros(batch_size, max_mention_num, max_len, dtype=torch.float32)
    entity_map = torch.zeros(batch_size, max_entity_num, max_mention_num, dtype=torch.float)
    hts = torch.zeros(batch_size, max_ht_num, 2, dtype=torch.int64)
    relations = torch.zeros(batch_size, max_ht_num, relation_num, dtype=torch.float32)
    relation_mask = torch.zeros(batch_size, max_ht_num, dtype=torch.bool)
    type_pair = torch.full((batch_size, max_ht_num), fill_value=36, dtype=torch.long)
    # type_mask = torch.zeros(batch_size, max_ht_num, relation_num, dtype=torch.bool)

    indexes = []
    titles = []
    all_label2in_train = []
    all_test_idxs = []

    graphs = []

    for idx, b in enumerate(batch):
        input_ids[idx, :len(b['input_id'])] = b['input_id']
        input_mask[idx, :len(b['input_id'])] = 1
        mention_map[idx, :b['mention_map'].shape[0], :b['mention_map'].shape[1]] = b['mention_map']
        entity_map[idx, :b['entity_map'].shape[0], :b['entity_map'].shape[1]] = b['entity_map']
        hts[idx, :len(b['hts'])] = b['hts']

        relations[idx, :len(b['relations'])] = b['relations']
        relation_mask[idx, :len(b['relations'])] = True
        type_pair[idx, :len(b['hts'])] = b['type_pair']
        # type_mask[idx, :len(b['relations'])] = b['type_mask']

        # for test
        titles.append(b['title'])
        indexes.append(b['index'])
        all_test_idxs.append(b['hts'].tolist())
        all_label2in_train.append(b['label2in_train'])
        graphs.append(b['graph'])

    return dict(input_id=input_ids,
                input_mask=input_mask,
                mention_map=mention_map,
                entity_map=entity_map,
                hts=hts,
                relations=relations,
                relation_mask=relation_mask,
                # type_mask=type_mask,
                type_pair=type_pair,

                # test
                titles=titles,
                indexes=indexes,
                labels=all_label2in_train,
                all_test_idxs=all_test_idxs,
                graphs=graphs)


class DataModule(LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = my_dataset(args.data_path, f"train")
        self.val_dataset = my_dataset(args.data_path, f'dev')
        self.test_dataset = my_dataset(args.data_path, f'test')

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, num_workers=self.num_workers,
                                      batch_size=self.batch_size, collate_fn=get_batch)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size * 4, collate_fn=get_batch,
                                    num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size * 4, collate_fn=get_batch,
                                     num_workers=self.num_workers)
        return test_dataloader
