from model.img_show import show_img_w_prediction
from data_handler.data_loader import load_datasets, get_dataloaders, get_id2label
from model.train import start_training
from model.def_model import save_model, get_model, get_img_processor
from settings import MODEL_PATH
from evaluation.evaluate_test_data import evaluate_on_test_data

CATEGORIES_PATH = 'data_handler/categories.json'

def main():
    print('---Menu list---')
    print('Type: 1 to train the model')
    print('Type: 2 to use the model')
    choice = input()
    if choice == '1':
        image_processor = get_img_processor()
        train_dataset, val_dataset, test_dataset = load_datasets(image_processor)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(image_processor, train_dataset, val_dataset, test_dataset)
        trained_model = start_training(train_dataloader, val_dataloader, test_dataloader, get_id2label(train_dataset))
        save_model(trained_model)
        evaluate_on_test_data(trained_model, test_dataloader, test_dataset)
    elif choice == '2':
        image_processor = get_img_processor()
        model = get_model(MODEL_PATH)
        show_img_w_prediction(image_processor, model, CATEGORIES_PATH)
    else:
        print('Input was not 1 or 2.')

if __name__ == '__main__':
    main()
