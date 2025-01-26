import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image

import random

import re

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import unicodedata

# SEED DEFINITION FUNCTION ------------------------------------------------------------------------------------------------------------

def set_random_seed(seed=42):
    
    """
    Sets the random seed for reproducibility across PyTorch, CUDA, NumPy, and Python's random module.
    
    Args:
        seed (int, optional): The seed value to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# IMAGES LOADER -----------------------------------------------------------------------------------------------------------------------

def load_image(image_path, transform):
    
    """
    Loads an image from the specified path, applies a given transformation, and returns the transformed image as a tensor.

    Args:
        image_path (str): The file path to the image to be loaded.
        transform (callable): A transformation function or pipeline that will be applied to the image. 

    Returns:
        The transformed image.
    """

    img = Image.open(image_path)
    transformed_img = transform(img)
    return transformed_img

# WORDS COUNTER FROM PANDAS FATAFRAME COLUMN ------------------------------------------------------------------------------------------

def column_words_counter(dataframe_column):

    """
    Counts the frequency of individual words in a specified column of a pandas DataFrame, where each entry in the column contains a list of words.

    Args:
        dataframe_column (pandas.Series): The DataFrame column.. Each entry in this column must be a list of words.

    Returns:
        dict: A dictionary where the keys are unique words from the lists in the specified column, and the values are their respective counts.
    """

    words_count = {}
    
    for words_list in dataframe_column:

        for word in words_list:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1
    
    return words_count


# TEXT STANDARIZER --------------------------------------------------------------------------------------------------------------------

def standardize_text(text):
    
    """
    Standardizes a given text by converting it to lowercase, removing diacritics, and replacing non-alphanumeric characters with spaces.

    Args:
        text (str): The input text to be standardized.

    Returns:
        str: The standardized text with all characters in lowercase, diacritics removed, and special characters replaced with spaces.
    """

    text = text.lower()

    text = unicodedata.normalize('NFD', text)
    
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])

    text = re.sub(r'[^a-z0-9 ]', ' ', text)

    return text


# WORDS SPLITTER ----------------------------------------------------------------------------------------------------------------------

def split_text(text, unique_words=False):

    """
    Splits a given text into a list of words and optionally returns only unique words.

    Args:
        text (str): The input text to be split into words.
        unique_words (bool, optional): If True, returns a set of unique words instead of a list. Defaults to False.

    Returns:
        list or set: A list of words if unique_words is False, otherwise a set of unique words.
    """

    words = text.split()

    if unique_words:
        words = set(words)
    
    return words


# RANDOM IMAGE VIEWER -----------------------------------------------------------------------------------------------------------------

def show_random_images(df_column, num_images=30):

    """
    Displays a random selection of images from a specified column of a DataFrame.

    Args:
        df_column (pandas.Series): The DataFrame column containing image tensors.
        num_images (int, optional): The number of images to display. Defaults to 30.

    Returns:
        None: Displays a grid of images using matplotlib.
    """

    sample_images = df_column.sample(n=min(num_images, len(df_column)), random_state=random.randint(0, 10000)).values
    
    rows, cols = 3, 10  
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, img_tensor in enumerate(sample_images):
        
        if i >= len(axes):
            break
        
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()

    plt.show()


# TOKENS CONVERTER TO INDICES ---------------------------------------------------------------------------------------------------------

def tokens_to_padded_indices(datafame_column, tokens_limit, padding_value=0, hash_limit=None):

    """
    Converts a list of tokens into padded indices, where each list of tokens is hashed and truncated or padded
    to a specified length.

    Args:
        dataframe_column (pandas.Series): A DataFrame column containing lists of tokens to be converted.
        tokens_limit (int): The maximum number of tokens (indices) for each list.
        padding_value (int, optional): The value to use for padding when the token list is shorter than the tokens_limit. Defaults to 0.
        hash_limit (int, optional): The limit for the hash value modulo operation. If None, no modulo is applied.

    Returns:
        torch.Tensor: A tensor containing the padded indices for each token list.
    """
    
    processed_lists = []
    
    for list in datafame_column:
        
        if hash_limit:
            indices = [hash(token) % hash_limit for token in list]
        else:
            indices = [hash(token) for token in list]
        
        if len(indices) > tokens_limit:
            processed_lists.append(indices[:tokens_limit])
        else:
            processed_lists.append(indices + [padding_value] * (tokens_limit - len(indices)))
    
    tensor = torch.tensor(processed_lists, dtype=torch.long)
    
    return tensor


# DATALOADERS CREATOR -----------------------------------------------------------------------------------------------------------------

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, seed=42):

    """
    Creates PyTorch DataLoader objects for training, validation, and test datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset to be used for training.
        val_dataset (torch.utils.data.Dataset): The dataset to be used for validation.
        test_dataset (torch.utils.data.Dataset): The dataset to be used for testing.
        seed (int, optional): The seed for random number generation, which ensures reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the DataLoader objects for training, validation, and test datasets.
    """

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# EPOCH TRAINER -----------------------------------------------------------------------------------------------------------------------

def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, criterion, optimizer, l1_lambda=None, scheduler=None):

    """
    Performs one epoch of training for the given model, calculating the loss and accuracy, 
    and updating the model's weights using backpropagation.

    Args:
        model (nn.Module): The neural network model to be trained.
        device (torch.device): The device (CPU or GPU) where the model and data are located.
        train_loader (DataLoader): The DataLoader providing the training data.
        criterion: The loss function to compute the error between predicted and true values.
        optimizer: The optimizer used to update the model's parameters.
        l1_lambda (float, optional): A regularization parameter for L1 regularization. Defaults to None.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): A learning rate scheduler to adjust the learning rate. Defaults to None.

    Returns:
        tuple: A tuple containing the average training loss, average training accuracy, and current learning rate (if scheduler is used).
    """

    model.train()
    
    avg_train_loss = 0.
    avg_train_acc = 0.

    for batch_idx, (batch_data_1, batch_data_2, batch_data_3, batch_target) in enumerate(train_loader):
        
        batch_data_1 = batch_data_1.to(device)
        batch_data_2 = batch_data_2.to(device)
        batch_data_3 = batch_data_3.to(device)
        batch_target =  batch_target.to(device)
        
        #Forward pass
        predicted_output = model(batch_data_1, batch_data_2, batch_data_3)
        loss = criterion(predicted_output, batch_target)
        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Losses and accuracy calculations
        avg_train_loss += (loss.item() * len(batch_target) / len(train_loader.dataset))
        
        _, predicted_index = predicted_output.max(1)
        correct_predictions = predicted_index.eq(batch_target).sum().item()
        avg_train_acc += 100. * correct_predictions / len(train_loader.dataset)    

    # Scheduler step
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        return avg_train_loss, avg_train_acc, current_lr
    
    else:
        return avg_train_loss, avg_train_acc


# MODEL EVALUATOR IN VALIDATION PHASE--------------------------------------------------------------------------------------------------

def eval_epoch(model: nn.Module, device: torch.device, val_loader: DataLoader, criterion, check_incorrects = False, confusion_matrix = False):

    """
    Evaluates the model on the validation dataset, calculating the loss and accuracy. Optionally prints incorrect predictions.

    Args:
        model (nn.Module): The neural network model to be evaluated.
        device (torch.device): The device (CPU or GPU) where the model and data are located.
        val_loader (DataLoader): The DataLoader providing the validation data.
        criterion: The loss function used to compute the error between predicted and true values.
        check_incorrects (bool, optional): If True, prints the incorrect predictions. Defaults to False.
        confusion_matrix (bool, optional): If True, computes and prints the confusion matrix of predictions.
        

    Returns:
        tuple: A tuple containing the average validation loss and average validation accuracy.
    """

    model.eval()

    avg_val_loss = 0.
    avg_val_acc = 0.

    if confusion_matrix:
        confusion = {
            0: {0: 0, 1: 0, 2: 0},
            1: {0: 0, 1: 0, 2: 0},
            2: {0: 0, 1: 0, 2: 0}
        }

    with torch.no_grad():
        
        for batch_data_1, batch_data_2, batch_data_3, batch_target in val_loader:

            batch_data_1 = batch_data_1.to(device)
            batch_data_2 = batch_data_2.to(device)
            batch_data_3 = batch_data_3.to(device)
            batch_target = batch_target.to(device)
            
            predicted_output = model(batch_data_1, batch_data_2, batch_data_3)
            loss = criterion(predicted_output, batch_target)
            
            avg_val_loss += (loss.item() * len(batch_target) / len(val_loader.dataset))
            
            _, predicted_index = predicted_output.max(1)
            correct_predictions = predicted_index.eq(batch_target).sum().item()
            avg_val_acc += 100. * correct_predictions / len(val_loader.dataset)

            if check_incorrects:
                incorrect_preds = predicted_index.ne(batch_target)
                for i, is_incorrect in enumerate(incorrect_preds):
                    if is_incorrect:
                        print(f"Índice predicho: {predicted_index[i].item()}, Valor real: {batch_target[i].item()}")
    
            if confusion_matrix:
                for i in range(len(batch_target)):
                    predicted_class = predicted_index[i].item()
                    true_class = batch_target[i].item()
                    confusion[predicted_class][true_class] += 1

    if confusion_matrix:
        confusion_df = pd.DataFrame(confusion)
        print("Matriz de Confusión:")
        print(confusion_df)

    return avg_val_loss, avg_val_acc


# ACCURACY AND LOSS CURVES PLOTTER ----------------------------------------------------------------------------------------------------

def plot_training_curves(train_losses, num_epochs, train_accs = None, val_losses = None, val_accs = None, test_acc = None):

    """
    Plots the training and validation loss/accuracy curves over the specified number of epochs.
    
    Args:
        train_losses (list): A list containing the training loss for each epoch.
        num_epochs (int): The number of epochs.
        train_accs (list, optional): A list containing the training accuracy for each epoch. Defaults to None.
        val_losses (list, optional): A list containing the validation loss for each epoch. Defaults to None.
        val_accs (list, optional): A list containing the validation accuracy for each epoch. Defaults to None.
        test_acc (float, optional): The accuracy on the test set, displayed as a horizontal line. Defaults to None.
    
    Returns:
        None: Displays the plots for training and validation loss/accuracy curves.
    """

    plt.style.use("ggplot")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    
    if val_losses is not None:
        plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    
    plt.legend()

    if train_accs is not None or val_accs is not None or test_acc is not None:
        
        plt.subplot(1, 2, 2)
        
        if train_accs is not None:
            plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
        
        if val_accs is not None:
            plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
        
        if test_acc is not None:
            plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
        
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        
        plt.legend()
    
    plt.tight_layout()
    
    plt.show()

# MODEL EVALUATOR IN TESTING PHASE ----------------------------------------------------------------------------------------------------

def evaluate_model(model: nn.Module, device: torch.device, test_loader: DataLoader, confusion_matrix = False):

    """
    Evaluates the model on the test dataset and optionally prints a confusion matrix.

    Args:
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to perform computations on.
        test_loader (DataLoader): The DataLoader for the test dataset.
        confusion_matrix (bool, optional): If True, prints the confusion matrix. Defaults to False.

    Returns:
        float: The average test accuracy.
    """

    model.eval()

    avg_test_acc = 0.

    if confusion_matrix:
        confusion = {
            0: {0: 0, 1: 0, 2: 0},
            1: {0: 0, 1: 0, 2: 0},
            2: {0: 0, 1: 0, 2: 0}
        }
    
    with torch.no_grad():
        
        for batch_data_1, batch_data_2, batch_data_3, batch_target in test_loader:

            batch_data_1 = batch_data_1.to(device)
            batch_data_2 = batch_data_2.to(device)
            batch_data_3 = batch_data_3.to(device)
            batch_target = batch_target.to(device)
            
            predicted_output = model(batch_data_1, batch_data_2, batch_data_3)
            
            _, predicted_index = predicted_output.max(1)
            correct_predictions = predicted_index.eq(batch_target).sum().item()
            avg_test_acc += 100. * correct_predictions / len(test_loader.dataset)

            if confusion_matrix:
                for i in range(len(batch_target)):
                    predicted_class = predicted_index[i].item()
                    true_class = batch_target[i].item()
                    confusion[predicted_class][true_class] += 1

    if confusion_matrix:
        confusion_df = pd.DataFrame(confusion)
        print("Matriz de Confusión:")
        print(confusion_df)

    return avg_test_acc

# -------------------------------------------------------------------------------------------------------------------------------------

