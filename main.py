from trainer.img_show import show_img_w_prediction, show_model_prediction
from trainer.data_loader import load_datasets, get_dataloaders, get_id2label
from trainer.train import start_training
from trainer.model import save_model, get_model, get_img_processor
from trainer.settings import CHECKPOINT, MODEL_PATH
from data_handler.data_split import split_dataset
from data_handler.data_labeler import extend_annotations

def main():
    print("---Menu list---")
    print("Type: 1 to train the model")
    print("Type: 2 to use the model")
    print("Type: 3 to split dataset")
    print("Type: 4 to extend the labels")
    choice = input()
    if choice == "1":
        image_processor = get_img_processor()
        pre_model = get_model(CHECKPOINT)
        # show_img_w_prediction(image_processor, pre_model)
        train_dataset, val_dataset, test_dataset = load_datasets(image_processor)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset)
        trained_model = start_training(train_dataloader, val_dataloader, get_id2label(train_dataset))
        #show_model_prediction(test_dataset, image_processor, trained_model)
        save_model(trained_model)
    elif choice == "2":
        image_processor = get_img_processor()
        model = get_model(MODEL_PATH)
        show_img_w_prediction(image_processor, model)
    elif choice == "3":
        split_dataset()
    elif choice == "4":
        extend_annotations()
    else:
        print("Input was not 1, 2, 3 or 4.")

if __name__ == "__main__":
    main()
