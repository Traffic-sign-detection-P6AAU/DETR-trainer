import os
from torch import DoubleTensor
import torchvision
from torch.utils.data import DataLoader
from settings import DATASET_PATH, BATCH_SIZE, NUM_WORKERS
from torch.utils.data.sampler import WeightedRandomSampler
# settings
ANNOTATION_FILE_NAME = '_annotations.coco.json'
TRAIN_DIRECTORY = os.path.join(DATASET_PATH, 'train')
VAL_DIRECTORY = os.path.join(DATASET_PATH, 'valid')
TEST_DIRECTORY = os.path.join(DATASET_PATH, 'test')

def load_datasets(image_processor):
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(
            self,
            image_directory_path: str,
            image_processor,
            train: bool
        ):
            annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
            super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
            self.image_processor = image_processor

        def __getitem__(self, idx):
            images, annotations = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            annotations = {'image_id': image_id, 'annotations': annotations}
            encoding = self.image_processor(images=images, annotations=annotations, return_tensors='pt')
            pixel_values = encoding['pixel_values'].squeeze()
            target = encoding['labels'][0]

            return pixel_values, target

    train_dataset = CocoDetection(
        image_directory_path=TRAIN_DIRECTORY,
        image_processor=image_processor,
        train=True)
    val_dataset = CocoDetection(
        image_directory_path=VAL_DIRECTORY,
        image_processor=image_processor,
        train=False)
    test_dataset = CocoDetection(
        image_directory_path=TEST_DIRECTORY,
        image_processor=image_processor,
        train=False)

    print('Number of training examples:', len(train_dataset))
    print('Number of validation examples:', len(val_dataset))
    print('Number of test examples:', len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset):
    def collate_fn(batch):
        # DETR authors employ various image sizes during training, making it not possible 
        # to directly batch together images. Hence they pad the images to the biggest 
        # resolution in a given batch, and create a corresponding binary pixel_mask 
        # which indicates which pixels are real/which are padding
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors='pt')
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
    
    train_dataloader = make_dataloader(train_dataset, collate_fn)
    train_dataloader.shuffle = True
    val_dataloader = make_dataloader(val_dataset, collate_fn)
    test_dataloader = make_dataloader(test_dataset, collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

def make_dataloader(dataset, cool_fn, use_sampler=False):
    sampler = None
    if use_sampler:
        # https://stackoverflow.com/questions/60812032/using-weightedrandomsampler-in-pytorch
        num_cats = len(dataset.coco.cats)
        num_imgs = len(dataset.coco.imgs)
        count = [0]*num_cats
        for ann in dataset.coco.imgToAnns.values():
            for a in ann:
                count[a['category_id']] += 1
        weights = []
        for i in range(num_cats):
            if count[i] == 0:
                print('{} (id: {}) has no samples'.format(dataset.coco.cats[i]['name'], i))
                weights.append(0)
            else:
                weights.append(1/count[i])
        print('\n') # to separate the output from the previous print
        sampler = WeightedRandomSampler(weights=DoubleTensor(weights), num_samples=num_imgs, replacement=True)

    return DataLoader(
        dataset=dataset,
        collate_fn=cool_fn,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        sampler=sampler)

def get_id2label(train_dataset):
    categories = train_dataset.coco.cats
    return {k: v['name'] for k,v in categories.items()}
