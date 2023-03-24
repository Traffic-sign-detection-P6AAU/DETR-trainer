from trainer.load_pre_trained import load_dataset, show_img_from_pre_dataset
from trainer.data_loader import load_datasets, get_dataloaders

def main():
    image_processor, preModel = load_dataset()
    show_img_from_pre_dataset(image_processor, preModel)
    train_dataset, val_dataset, test_dataset = load_datasets(image_processor)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset)

if __name__ == "__main__":
    main()
