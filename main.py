from trainer.load_pre_trained import load_dataset, show_img_from_pre_dataset
from trainer.data_loader import load_datasets, get_dataloaders, get_id2label
from trainer.train import start_training, save_model
from trainer.test_model import show_model_prediction

def main():
    image_processor, pre_model = load_dataset()
    #show_img_from_pre_dataset(image_processor, pre_model)
    train_dataset, val_dataset, test_dataset = load_datasets(image_processor)
   
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset)
    trained_model = start_training(train_dataloader, val_dataloader, get_id2label(train_dataset))
    save_model(trained_model)
    #show_model_prediction(test_dataset, image_processor, trained_model)

if __name__ == "__main__":
    main()
