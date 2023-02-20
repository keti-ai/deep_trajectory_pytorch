
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.base_dataset import BaseDataSet

from torch.utils.data.sampler import BatchSampler, SequentialSampler

import time
class BaseDataLoader(DataLoader):
    def __init__(self, dataset=BaseDataSet, batch_size=1, shuffle=False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None, *, prefetch_factor=2,persistent_workers=False):
        super().__init__(dataset=dataset,persistent_workers=persistent_workers,batch_size=batch_size,sampler=sampler,drop_last=drop_last)
        # self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
        self.timeout = timeout
        self.collate_fn = collate_fn
        # self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        # self.batch_sampler = batch_sampler
        # self.sampler = samplers
        self.shuffle = shuffle
        # self.batch_size = batch_size

def base_collate_fn(samples):
    in_=[]
    label_=[]
    for i,j in samples:
        in_.append(i)
        label_.append(j)
    return {"input":in_,"label":label_}


class BaseSampler():
    def __init__(self):
        pass
    def __len__(self):
        pass

class Trainer():
    def __init__(self,model):
        self.model = model()
        pass
    def train(self):
        print("do train")
        basedataset = BaseDataSet()
        dataloader = BaseDataLoader(
            dataset=basedataset,
            collate_fn=base_collate_fn,
            batch_size=1,
            sampler=BatchSampler(
                SequentialSampler(basedataset),
                batch_size=1,
                drop_last=False
            )
        )

        for i_batch,feed_dict in enumerate(tqdm(dataloader)):
            print("\n")
            print("idx : ",i_batch)
            print(feed_dict['input'])
            print(feed_dict['label'])
            time.sleep(0.25)






