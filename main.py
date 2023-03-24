from trainer.load_pre_trained import load_dataset, show_img_from_dataset

def main():
    image_processor, model = load_dataset()
    show_img_from_dataset(image_processor, model)


if __name__ == "__main__":
    main()
