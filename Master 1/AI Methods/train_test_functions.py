"""
Module to define the train loop, the test loop and a global loop who combine both
"""

import torch
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def global_loop(model, model_name, train, test):
    """
    Use training and testing functions on the model.
    Save as csv the accuracy and the loss on the test dataset in results folder for each epoch of each loop.
    Save the last model calculated in results folder.

    Parameters
    ----------
    model : nn.modules.Module
    model_name : string
    train : DataLoader
    test : DataLoader

    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    EPOCHS = 10
    # define the number of loop to calculate accuracies and losses mean per model
    loop_by_model = 5

    model.to(device)

    torch.save(model.state_dict(), "./results/reset_model")

    accuracy_matrix = np.zeros((loop_by_model, EPOCHS))
    loss_matrix = np.zeros((loop_by_model, EPOCHS))

    for loop in range(loop_by_model):

        model.load_state_dict(torch.load("./results/reset_model"))

        # Training phase.
        for epoch in range(EPOCHS):

            training_acc, training_loss = training(model, train, optimizer, criterion)
            test_acc, test_loss = testing(model, test, criterion)

            print(
                "{} --- Epoch{}/{} --- train loss : {} | train acc : {} | test loss : {} | test acc {}".format(
                    model_name,
                    epoch + 1,
                    EPOCHS,
                    training_loss,
                    training_acc,
                    test_loss,
                    test_acc,
                )
            )

            accuracy_matrix[loop, epoch] = test_acc
            loss_matrix[loop, epoch] = test_loss

    np.savetxt(
        "./results/{}_acc.csv".format(model_name), accuracy_matrix, delimiter=","
    )
    np.savetxt("./results/{}_loss.csv".format(model_name), loss_matrix, delimiter=",")
    torch.save(model.state_dict(), "./results/{}".format(model_name))


def training(model, train, optimizer, criterion):
    """
    Train the model with the train dataset

    Parameters
    ----------
    model : nn.modules.Module
    train : DataLoader
    optimizer : orch.optim.Optimizer
    criterion : nn.modules.loss

    Returns
    ---------------
    train accuracy : float
    train loss : float
    """
    train_running_loss = 0
    correct_train = 0

    for inputs, labels in train:
        inputs = inputs.to(device)
        labels = labels.to(device)

        """ for every mini-batch during the training phase, we typically want to explicitly set the gradients 
                to zero before starting to do backpropragation """
        optimizer.zero_grad()

        # Start the forward pass
        prediction = model(inputs)

        loss = criterion(prediction, labels)

        # do backpropagation and update weights with step()
        loss.backward()
        optimizer.step()

        # find the maximum along the rows, use dim=1 to torch.max(), equivalent top-1 error
        _, predicted_outputs = torch.max(prediction.data, 1)

        # update the running corrects
        correct_train += (predicted_outputs == labels).float().sum().item()

        train_running_loss += loss.data.item() * inputs.shape[0]

    return correct_train / len(train.dataset), train_running_loss / len(train.dataset)


def testing(model, test, criterion):
    """
    Test the model on the test dataset

    Parameters
    ----------
    model : nn.modules.Module
    test : DataLoader
    criterion : nn.modules.loss

    Returns
    ---------------
    test accuracy : float
    test loss : float
    """
    test_running_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs_test, labels_test in test:

            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # Make predictions.
            prediction = model(inputs_test)

            # Calculation of the loss
            loss = criterion(prediction, labels_test)

            # Retrieve predictions indexes.
            _, predicted_class = torch.max(prediction.data, 1)

            # Compute number of correct predictions.
            correct += (predicted_class == labels_test).float().sum().item()

            test_running_loss += loss.data.item() * inputs_test.shape[0]

    return correct / len(test.dataset), test_running_loss / len(test.dataset)


def torch_loop(model, model_name, train, test):
    """
    Use training and testing functions on the torch resnet model.
    Save as csv the accuracy and the loss on the test dataset in results folder.
    Save the model calculated in results folder.

    Parameters
    ----------
    model : nn.modules.Module
    model_name : string
    train : DataLoader
    test : DataLoader
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    EPOCHS = 10

    model.to(device)

    accuracy_matrix = np.zeros(EPOCHS)
    loss_matrix = np.zeros(EPOCHS)

    # Training phase.
    for epoch in range(EPOCHS):

        training_acc, training_loss = training(model, train, optimizer, criterion)
        test_acc, test_loss = testing(model, test, criterion)

        print(
            "{} --- Epoch{}/{} --- train loss : {} | train acc : {} | test loss : {} | test acc {}".format(
                model_name,
                epoch + 1,
                EPOCHS,
                training_loss,
                training_acc,
                test_loss,
                test_acc,
            )
        )

        accuracy_matrix[epoch] = test_acc
        loss_matrix[epoch] = test_loss

    np.savetxt(
        "./results/{}_acc.csv".format(model_name), accuracy_matrix, delimiter=","
    )
    np.savetxt("./results/{}_loss.csv".format(model_name), loss_matrix, delimiter=",")
    torch.save(model.state_dict(), "./results/{}".format(model_name))
