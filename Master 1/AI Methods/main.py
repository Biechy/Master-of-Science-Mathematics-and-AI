# open sphinx documentation on your web browser
from analysis import (
    Compare_OptionA_and_B,
    Compare_ResNet_and_TorchResNet,
    Compare_ResNet_and_CNN,
)
from train_test_functions import torch_loop
from models import TorchResNet50, TorchResNet34, TorchResNet18
from train_test_functions import global_loop
from models import (
    ResNet56A,
    ResNet44A,
    ResNet32A,
    ResNet20A,
    ResNet56B,
    ResNet44B,
    ResNet32B,
    ResNet20B,
    CNN56,
    CNN44,
    CNN32,
    CNN20,
)
from data import LoadCIFAR10
import webbrowser
import os

file_name = "./docs/_build/html/index.html"
path = os.path.realpath(file_name)
webbrowser.open_new_tab(path)


train, test = LoadCIFAR10(batch_size=32)


"""
All of the result (model, loss, accuracy, plot, etc) are in the folder results
You can pass from here...
"""

models = [
    ResNet56A(),
    ResNet44A(),
    ResNet32A(),
    ResNet20A(),
    ResNet56B(),
    ResNet44B(),
    ResNet32B(),
    ResNet20B(),
    CNN56(),
    CNN44(),
    CNN32(),
    CNN20(),
]
models_str = [
    "ResNet56A",
    "ResNet44A",
    "ResNet32A",
    "ResNet20A",
    "ResNet56B",
    "ResNet44B",
    "ResNet32B",
    "ResNet20B",
    "CNN56",
    "CNN44",
    "CNN32",
    "CNN20",
]


for model, model_name in zip(models, models_str):
    global_loop(model, model_name, train, test)


models = [TorchResNet50(), TorchResNet34(), TorchResNet18()]
models_str = ["TorchResNet50", "TorchResNet34", "TorchResNet18"]


for model, model_name in zip(models, models_str):
    torch_loop(model, model_name, train, test)
"""...to here"""


Compare_OptionA_and_B()
Compare_ResNet_and_TorchResNet()
Compare_ResNet_and_CNN()
