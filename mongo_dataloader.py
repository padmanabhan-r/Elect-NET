from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class DatasetDB(Dataset):
    def __init__(self, db_name='ElecNet', col_name='ImgColl', set_name='TR'):
        client = MongoClient('localhost', 27017)
        db = client[db_name]
        self.col = db[col_name]
        self.set = set_name
        self.examples = list(self.col.find({'flag':self.set}, {'image_bytes': 0}))
        self.labels = self.get_category()
        self.trans_tensor = transforms.Compose([transforms.Resize(224),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(0,45),
                                                transforms.RandomVerticalFlip(0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    ])
        
    def __len__(self):
        return len(self.examples)

    def get_category(self):
        category = [e['category'] for e in self.examples]
        return {category_id: i for i, category_id in enumerate(sorted(list(set(category))))}

    def __getitem__(self, i):
        _id = self.examples[i]['_id']
        doc = self.col.find_one({'_id': _id})
        img = doc['image_bytes']
        cat = doc['category']
        width = doc['width']
        height = doc['height']
        try:
            image = Image.frombytes("RGB", (width, height), bytes(img))
        except ValueError as v:
            return (self.trans_tensor(Image.new("RGB",(224,224))), cat)
        return (self.trans_tensor(image), cat)