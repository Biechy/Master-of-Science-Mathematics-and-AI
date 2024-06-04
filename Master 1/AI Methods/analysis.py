"""
Module to download plot of comparaisons
refer to global_loop function in train_test_functions.py to see how named are encoded 
"""
import numpy as np
import matplotlib.pyplot as plt



def Compare_OptionA_and_B():
    ResNet56A_acc = np.loadtxt("./results/ResNet56A_acc.csv", delimiter=",")
    ResNet44A_acc = np.loadtxt("./results/ResNet44A_acc.csv", delimiter=",")
    ResNet32A_acc = np.loadtxt("./results/ResNet32A_acc.csv", delimiter=",")
    ResNet20A_acc = np.loadtxt("./results/ResNet20A_acc.csv", delimiter=",")
    ResNet56B_acc = np.loadtxt("./results/ResNet56B_acc.csv", delimiter=",")
    ResNet44B_acc = np.loadtxt("./results/ResNet44B_acc.csv", delimiter=",")
    ResNet32B_acc = np.loadtxt("./results/ResNet32B_acc.csv", delimiter=",")
    ResNet20B_acc = np.loadtxt("./results/ResNet20B_acc.csv", delimiter=",")

    ResNet56A_loss = np.loadtxt("./results/ResNet56A_loss.csv", delimiter=",")
    ResNet44A_loss = np.loadtxt("./results/ResNet44A_loss.csv", delimiter=",")
    ResNet32A_loss = np.loadtxt("./results/ResNet32A_loss.csv", delimiter=",")
    ResNet20A_loss = np.loadtxt("./results/ResNet20A_loss.csv", delimiter=",")
    ResNet56B_loss = np.loadtxt("./results/ResNet56B_loss.csv", delimiter=",")
    ResNet44B_loss = np.loadtxt("./results/ResNet44B_loss.csv", delimiter=",")
    ResNet32B_loss = np.loadtxt("./results/ResNet32B_loss.csv", delimiter=",")
    ResNet20B_loss = np.loadtxt("./results/ResNet20B_loss.csv", delimiter=",")

    matrices_acc = [ResNet56A_acc, ResNet44A_acc, ResNet32A_acc, ResNet20A_acc, ResNet56B_acc, ResNet44B_acc, ResNet32B_acc, ResNet20B_acc]
    matrices_loss = [ResNet56A_loss, ResNet44A_loss, ResNet32A_loss, ResNet20A_loss, ResNet56B_loss, ResNet44B_loss, ResNet32B_loss, ResNet20B_loss]
    matrices_str = ["ResNet56A", "ResNet44A", "ResNet32A", "ResNet20A", "ResNet56B", "ResNet44B", "ResNet32B", "ResNet20B"]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Accuracy of ResNet with shortcut A or B")
    for matrix, matrix_name in zip(matrices_acc, matrices_str):
        matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Loss of ResNet with shortcut A or B")
    for matrix, matrix_name in zip(matrices_loss, matrices_str):
        matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.savefig("./results/OptionAandB.png")


def Compare_ResNet_and_TorchResNet():
    ResNet56A_acc = np.loadtxt("./results/ResNet56A_acc.csv", delimiter=",")
    ResNet32A_acc = np.loadtxt("./results/ResNet32A_acc.csv", delimiter=",")
    ResNet20A_acc = np.loadtxt("./results/ResNet20A_acc.csv", delimiter=",")
    TorchResNet50_acc = np.loadtxt("./results/TorchResNet50_acc.csv", delimiter=",")
    TorchResNet34_acc = np.loadtxt("./results/TorchResNet34_acc.csv", delimiter=",")
    TorchResNet18_acc = np.loadtxt("./results/TorchResNet18_acc.csv", delimiter=",")

    ResNet56A_loss = np.loadtxt("./results/ResNet56A_loss.csv", delimiter=",")
    ResNet32A_loss = np.loadtxt("./results/ResNet32A_loss.csv", delimiter=",")
    ResNet20A_loss = np.loadtxt("./results/ResNet20A_loss.csv", delimiter=",")
    TorchResNet50_loss = np.loadtxt("./results/TorchResNet50_loss.csv", delimiter=",")
    TorchResNet34_loss = np.loadtxt("./results/TorchResNet34_loss.csv", delimiter=",")
    TorchResNet18_loss = np.loadtxt("./results/TorchResNet18_loss.csv", delimiter=",")

    matrices_acc = [ResNet56A_acc, ResNet32A_acc, ResNet20A_acc, TorchResNet50_acc, TorchResNet34_acc, TorchResNet18_acc]
    matrices_loss = [ResNet56A_loss, ResNet32A_loss, ResNet20A_loss, TorchResNet50_loss, TorchResNet34_loss, TorchResNet18_loss]
    matrices_str = ["ResNet56A", "ResNet32A", "ResNet20A", "TorchResNet50", "TorchResNet34", "TorchResNet18"]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Accuracy of ResNet vs TorchResNet")
    for matrix, matrix_name in zip(matrices_acc, matrices_str):
        if matrix_name != "TorchResNet50" and matrix_name != "TorchResNet34" and matrix_name != "TorchResNet18":
            matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Loss of ResNet vs TorchResNet")
    for matrix, matrix_name in zip(matrices_loss, matrices_str):
        matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.savefig("./results/ResNetandTorchResNet.png")



def Compare_ResNet_and_CNN():
    ResNet56A_acc = np.loadtxt("./results/ResNet56A_acc.csv", delimiter=",")
    ResNet44A_acc = np.loadtxt("./results/ResNet44A_acc.csv", delimiter=",")
    ResNet32A_acc = np.loadtxt("./results/ResNet32A_acc.csv", delimiter=",")
    ResNet20A_acc = np.loadtxt("./results/ResNet20A_acc.csv", delimiter=",")
    CNN56_acc = np.loadtxt("./results/CNN56_acc.csv", delimiter=",")
    CNN44_acc = np.loadtxt("./results/CNN44_acc.csv", delimiter=",")
    CNN32_acc = np.loadtxt("./results/CNN32_acc.csv", delimiter=",")
    CNN20_acc = np.loadtxt("./results/CNN20_acc.csv", delimiter=",")

    ResNet56A_loss = np.loadtxt("./results/ResNet56A_loss.csv", delimiter=",")
    ResNet44A_loss = np.loadtxt("./results/ResNet44A_loss.csv", delimiter=",")
    ResNet32A_loss = np.loadtxt("./results/ResNet32A_loss.csv", delimiter=",")
    ResNet20A_loss = np.loadtxt("./results/ResNet20A_loss.csv", delimiter=",")
    CNN56_loss = np.loadtxt("./results/CNN56_loss.csv", delimiter=",")
    CNN44_loss = np.loadtxt("./results/CNN44_loss.csv", delimiter=",")
    CNN32_loss = np.loadtxt("./results/CNN32_loss.csv", delimiter=",")
    CNN20_loss = np.loadtxt("./results/CNN20_loss.csv", delimiter=",")

    matrices_acc = [ResNet56A_acc, ResNet44A_acc, ResNet32A_acc, ResNet20A_acc, CNN56_acc, CNN44_acc, CNN32_acc, CNN20_acc]
    matrices_loss = [ResNet56A_loss, ResNet44A_loss, ResNet32A_loss, ResNet20A_loss, CNN56_loss, CNN44_loss, CNN32_loss, CNN20_loss]
    matrices_str = ["ResNet56A", "ResNet44A", "ResNet32A", "ResNet20A", "CNN56", "CNN44", "CNN32", "CNN20"]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Accuracy of ResNet vs CNN")
    for matrix, matrix_name in zip(matrices_acc, matrices_str):
        matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Loss of ResNet vs CNN")
    for matrix, matrix_name in zip(matrices_loss, matrices_str):
        matrix = np.mean(matrix, axis=0)
        plt.plot(matrix , label=str(matrix_name))
    plt.legend()

    plt.savefig("./results/ResNetvsCNN.png")