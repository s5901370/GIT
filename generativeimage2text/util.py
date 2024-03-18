import jsonlines
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import BertTokenizer
def write_log(args,im):
    with open(f"./log/{args['exp_name']}_bs{args['bs']}_lr{args['lr']}_im{im}_log.txt","a") as f:
        f.write(f"bs = {args['bs']}, num_epoch = {args['epoch']},\n")
        f.write(f"lr = {args['lr']}, wd = {args['wd']},\n")

def get_images(id, data_path, length, num_images,trsfm):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path+id
    images = []
    if length >=num_images:
        for i in range(num_images):
            # get last six images
            image_path = f"{path}-{length-num_images+i}-{length}.jpg"
            images.append(trsfm(Image.open(image_path).convert('RGB')))

    else:
        for i in range(length):
            # get last six images
            image_path = f"{path}-{i}-{length}.jpg"
            images.append(trsfm(Image.open(image_path).convert('RGB')))
        shape = images[0].shape
        for i in range(num_images-length):
            images.append(torch.zeros(shape))
        # for i in range(num_images-length):
        #     images.append()
    # Image.open(image_path).convert('RGB')
    images_stacked = torch.stack(images, dim=0)
    return images_stacked

def trsfm(image_size = 224,split = 'VALID'):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if split == 'TRAIN':
        ret = v2.Compose(
                [
                    v2.RandomResizedCrop(size=image_size, interpolation=InterpolationMode.BICUBIC),
                    v2.RandomAffine(degrees=0,translate=(0.2, 0.2), scale=(0.75, 1)),
                    v2.ColorJitter(brightness=0.5, contrast=0.5),
                    v2.ToTensor(),
                    v2.Normalize(mean, std)
                ]
            )
    else:
        ret = v2.Compose(
                [
                    v2.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                    ),
                    v2.ToTensor(),
                    v2.Normalize(mean, std)
                ]
            )
    return ret
class AITW_Dataset(Dataset):
    def __init__(self, jsonl_file, data_path,split,tokenizer,transform, num_images=6 ):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Read data from JSONL file
        with jsonlines.open(jsonl_file) as reader:
            size = sum(1 for i in reader)
        
        with jsonlines.open(jsonl_file) as reader:
            if split == 'TRAIN':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i < limit:
                        self.data.append(lines)
                    else:
                        break
            if split == 'VALID':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i >= limit:
                        self.data.append(lines)

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        images = get_images(item['id'],self.data_path,item['episode_length'],self.num_images,self.transform)

        max_text_len = 40 #from train.py
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        # print('get_data')
        # print(prefix)
        # print(target)
        # print(need_predict)
        # print(payload)
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data

def main():
    annotations_file = '/local/poyang/test/105_train.jsonl'
    data_path = '/data/cv/poyang/AITW/GoogleApps/'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = AITW_Dataset(annotations_file,data_path,'TRAIN',tokenizer,transform = trsfm())
    for i in range(5):
        print(dataset[i])
    # trainloader = DataLoader(dataset, batch_size=32, num_workers=2, collate_fn = dataset.collate_fn)
    # print(next(iter(trainloader))['caption_tokens'])
    # print(next(iter(trainloader))['need_predict'])
    # for i,data in enumerate(trainloader):
    #     if i == 1:
    #         break
    #     print(data)


if __name__ == '__main__':
    # main()
    # print_demo_images()