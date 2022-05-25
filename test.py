import torch
import numpy as np
from models.simple_nn import SimpleCNN
from skimage.io import imread, imsave
from skimage.transform import resize
from argparse import ArgumentParser

def run_inference(input_img_path, output_img_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load("./data/checkpoints/saved_model.pth"))

    input_img = imread(input_img_path)
    if input_img.shape != (32,32,3):
        input_img = resize(input_img, (32, 32))

    in_ten = torch.tensor(np.array([input_img])).permute(0, 3, 1, 2)
    import pdb; pdb.set_trace()
    outputs = model(in_ten)
    _, predicted = torch.max(outputs, 1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    print(f"Predicted class {classes[predicted[j]] for j in range(4)}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    args = parser.parse_args()

    run_inference(args.i, args.o)
