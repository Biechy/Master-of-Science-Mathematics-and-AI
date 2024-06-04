import streamlit as st
import base64
import requests
import webbrowser
from streamlit_timeline import timeline
import os
import matplotlib.pyplot as plt
import time
import streamlit as st
import numpy as np
from models import ResNet8, ResNet14, ResNet20, CNN8, CNN14, CNN20
from data import LoadCIFAR10_subset
from train_test_functions import training, testing
from torch.utils.data import DataLoader
from train_test_functions import global_loop
from torch import nn
import torch

st.set_page_config(layout="wide")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://imgs.search.brave.com/NM8CWFNYWF4jyTq4erhz0lrU0ZXqu6sOHCYC3aCvR6w/rs:fit:1200:1200:1/g:ce/aHR0cHM6Ly93YWxs/dXAubmV0L3dwLWNv/bnRlbnQvdXBsb2Fk/cy8yMDE2LzAzLzA5/LzM0MzY2NC1kaWdp/dGFsX2FydC1hYnN0/cmFjdC1taW5pbWFs/aXNtLWJsYWNrLTNE/LWxpbmVzLXNpbXBs/ZS5qcGc");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def displayPDF():
    # Opening file from file url
    paper_url = "https://arxiv.org/pdf/1512.03385.pdf"
    response = requests.get(paper_url)
    content = response.content
    base64_pdf = base64.b64encode(content).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="500" height="700" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


st.title("ResNet")

st.header("Introduction")


timeline_data = {
    "title": {
        "media": {"url": "", "caption": "", "credit": ""},
        "text": {
            "headline": "history of deep neural networks",
            "text": "<p>Deep neural networks were developed in the 1940s to model brain function, but their practical use was limited until a deep convolutional neural network won the ImageNet competition in 2012. Since then, they have been widely used for various tasks in artificial intelligence.</p>",
        },
    },
    "events": [
        {
            "media": {
                "url": "https://imgs.search.brave.com/TN8QlYSDh0VSUQl-mpOUwG_nGc-GK0MJ53HxdettkpA/rs:fit:905:225:1/g:ce/aHR0cHM6Ly90c2U0/Lm1tLmJpbmcubmV0/L3RoP2lkPU9JUC5I/aFVvVVBHQ1FpdlFK/ckxweFdfenVnSGFE/NCZwaWQ9QXBp",
                "caption": "Walter Pitts",
            },
            "start_date": {"year": "1940"},
            "text": {
                "headline": "Beginning",
                "text": "The work of neurophysiologist Warren McCulloch and logician Walter Pitts on neural network modeling laid the theoretical foundation for neural networks.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/rnp45N2yfdKL3Gr1GjVAYslkoos_5sfnzifxBNCGWYE/rs:fit:1200:821:1/g:ce/aHR0cDovL3N0YXRp/YzEuYnVzaW5lc3Np/bnNpZGVyLmNvbS9p/bWFnZS81YTY3ODhh/OWEyNDQ0NGUxM2E4/YjUyNTItMTY0My95/YW5uJTIwbGVjdW4u/anBn",
                "caption": "Yann LeCun",
                "credit": "",
            },
            "start_date": {"year": "1998"},
            "text": {
                "headline": "LeNet",
                "text": "However, it was not until the 1980s that neural networks began to be used for image recognition. Researcher Yann LeCun developed a convolutional neural network called LeNet to recognize handwritten characters on bank checks.  Now ResNet refers to the ResNet 5 network released in 1989.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/hy4Iv2ukO35-OQHVtIrqI8iNG8hHxkMzBE5NdoaEA7s/rs:fit:1200:900:1/g:ce/aHR0cHM6Ly9lMy4z/NjVkbS5jb20vMjEv/MDMvMTYwMHg5MDAv/c2t5bmV3cy1nZW9m/ZnJleS1oaW50b25f/NTMwOTMzMS5qcGc_/MjAyMTAzMTgxMTA5/NTU",
                "caption": "Geoffrey Hinton",
                "credit": "",
            },
            "start_date": {"year": "2012"},
            "text": {
                "headline": "The revival of convolutional neural networks",
                "text": "The research team led by Geoffrey Hinton used a deep convolutional neural network to win the ImageNet competition, which marked a turning point for deep neural networks and contributed to their growing popularity.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/brfTeVM6pGXKMIUNZGQ5H3QOADHJVvDQByV6tRT5BWo/rs:fit:900:848:1/g:ce/aHR0cHM6Ly93d3cu/Y2xpcGFydGtleS5j/b20vbXBuZ3MvbS8z/MS0zMTEzNDZfb3hm/b3JkLXVuaXZlcnNp/dHktbG9nby5wbmc",
                "caption": "",
                "credit": "",
            },
            "start_date": {"year": "2014"},
            "text": {
                "headline": "VGG",
                "text": "The VGG network was developed in 2014 by a group of researchers at Oxford University. The VGG network was particularly notable for its simple, symmetric structure, which consisted of a series of convolution layers with 3x3 size filters, followed by clustering layers, fully connected layers, and classification layers.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/LrTVOkmNdw0CcgbcKDPK7MlVfDFwvCKhjAQYZkGyC4w/rs:fit:474:225:1/g:ce/aHR0cHM6Ly90c2Uy/Lm1tLmJpbmcubmV0/L3RoP2lkPU9JUC55/NS05MFA0U2lndkRY/MzNHNjlwc2J3SGFI/YSZwaWQ9QXBp",
                "caption": "",
                "credit": "",
            },
            "start_date": {"year": "2015"},
            "text": {
                "headline": "GoogLeNet",
                "text": "The unique feature of GoogleNet is its 'Inception' architecture, which uses convolution and clustering blocks of different sizes in parallel to extract features at multiple spatial scales. This architecture has made it possible to significantly reduce the number of model parameters while maintaining high classification accuracy.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/zI6HVrn9WNaqsiJPqPoJuwTnfYx-bmhMUGxodxgUmsA/rs:fit:1200:1200:1/g:ce/aHR0cHM6Ly9uZXdz/LWNkbi5zb2Z0cGVk/aWEuY29tL2ltYWdl/cy9uZXdzMi9NaWNy/b3NvZnQtUmVkZXNp/Z25zLUl0cy1Mb2dv/LWZvci10aGUtRmly/c3QtVGltZS1pbi0y/NS1ZZWFycy1IZXJl/LUl0LUlzLTMucG5n",
                "caption": "",
                "credit": "",
            },
            "start_date": {"year": "2015"},
            "text": {
                "headline": "ResNet",
                "text": "The unique feature of ResNet is its architecture that uses residual blocks, which allow information to bypass the intermediate layers of the network. This technique, called 'skip connection', solved the back-propagation problem and improved the performance of the network for depths up to several hundred layers.",
            },
        },
        {
            "media": {
                "url": "https://imgs.search.brave.com/T8o6orR5B07yAP2FxaIzAt1pI-uO8zxV-KX9WndFeZA/rs:fit:560:320:1/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy90/aHVtYi80LzQyL0Nv/cm5lbGxfVW5pdmVy/c2l0eV9Mb2dvLnBu/Zy81MTJweC1Db3Ju/ZWxsX1VuaXZlcnNp/dHlfTG9nby5wbmc",
                "caption": "",
                "credit": "",
            },
            "start_date": {"year": "2016"},
            "text": {
                "headline": "DenseNet",
                "text": "(State of the art) <br> DenseNet uses dense connections between network layers, meaning that each layer is connected to all previous layers in a layer block. This structure allows for faster and more efficient propagation of information through the network, reducing the risk of vanishing gradients and improving overall performance.",
            },
        },
    ],
}

timeline(timeline_data, height=500)

col1, col2 = st.columns(2)
with col1:
    displayPDF()
with col2:
    st.write(
        "Our project is to study the *Deep Residual Learning for Image Recognition* paper that introduces the ResNet network written by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun, researchers at Microsoft. This new network is characterized by the idea of direct connection, which solves the problem of vanishing and exploding gradients and allows to create extremely deep networks. It win the ImageNet competition in 2015. Specifically, the ResNet-152 version achieved a classification error of 3.57%, which was a significant improvement over previous models.  So we're going to try to replicate that on our scale. This web page is accompanied by an article we wrote available by clicking on the button below and aims to complete it in an interactive way."
    )
    if st.button("Our Article About ResNet"):
        webbrowser.open_new("streamlit_ressources\ResNet_Cavalier_Biechy.pdf")

st.header("Creation of models")
col1, col2 = st.columns(2)
with col1:
    # Affichage des images et de la barre de sélection
    selected_model = st.slider(
        "Depth of the convolutionel network",
        min_value=8,
        max_value=20,
        value=14,
        step=6,
    )
    model_type = st.selectbox("Choose the type of your model", ("ResNet", "CNN"))


with col2:
    # Création d'une liste contenant les URL des images
    model_images = [
        [
            "streamlit_ressources\ResNet8.jpeg",
            "streamlit_ressources\ResNet14.jpeg",
            "streamlit_ressources\ResNet20.jpeg",
        ],
        [
            "streamlit_ressources\CNN8.jpeg",
            "streamlit_ressources\CNN14.jpeg",
            "streamlit_ressources\CNN20.jpeg",
        ],
    ]
# Initialiser le booléen pour afficher ou masquer la barre latérale
show_sidebar = False

# Ajouter un bouton pour afficher ou masquer la barre latérale
if st.button("View model"):
    show_sidebar = not show_sidebar

# Affichage de l'image sélectionnée
if selected_model == 8 and model_type == "ResNet":
    if show_sidebar:
        st.sidebar.image(model_images[0][0])
    model = ResNet8()
    name = "ResNet8"

elif selected_model == 14 and model_type == "ResNet":
    if show_sidebar:
        st.sidebar.image(model_images[0][1])
    model = ResNet14()
    name = "ResNet14"
elif selected_model == 20 and model_type == "ResNet":
    if show_sidebar:
        st.sidebar.image(model_images[0][2])
    model = ResNet20()
    name = "ResNet20"
elif selected_model == 8 and model_type == "CNN":
    if show_sidebar:
        st.sidebar.image(model_images[1][0])
    model = CNN8()
    name = "CNN8"
elif selected_model == 14 and model_type == "CNN":
    if show_sidebar:
        st.sidebar.image(model_images[1][1])
    model = CNN14()
    name = "CNN14"
elif selected_model == 20 and model_type == "CNN":
    if show_sidebar:
        st.sidebar.image(model_images[1][2])
    model = CNN20()
    name = "CNN20"


def train_model(model, name):
    st.write("Training model")
    progress_bar_container = st.empty()
    progress_text_container = st.empty()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    acc = []

    EPOCHS = 5
    fig, ax = plt.subplots()

    for epoch in range(EPOCHS):

        training_acc, training_loss = training(model, train, optimizer, criterion)
        test_acc, test_loss = testing(model, test, criterion)

        progress_percent = (epoch + 1) / EPOCHS
        progress_bar_container.progress(progress_percent)
        progress_text_container.write(f"{int(100*progress_percent)}%")
        time.sleep(1)

        acc.append(test_acc)

    st.write("Modèle entraîné avec succès !")

    np.savetxt("./streamlit_res/{}_acc.csv".format(name), acc, delimiter=",")

    ax.set_title("Accuracy according to the epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.plot(acc)

    fig.set_size_inches(8, 6)

    st.pyplot(fig)

    st.write(
        "Nombre de paramètre {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000
        )
    )


def load_res():
    text_files = []
    for file_name in os.listdir("streamlit_res"):
        text_files.append(file_name)

    fig, ax = plt.subplots()

    for text_file in text_files:
        data = np.loadtxt(os.path.join("streamlit_res", text_file))
        ax.plot(data, label=text_file)

        ax.legend()
        ax.set_title("Accuracy according to the epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")

    st.pyplot(fig)


train, test = LoadCIFAR10_subset(batch_size=32, subset=5000)

if st.button("Train model"):
    train_model(model, name)

if st.button("Print accuracy of every model trained so far"):
    load_res()
