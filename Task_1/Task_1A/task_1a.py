'''
*****************************************************************************************
*
*                ===============================================
*               GeoGuide(GG) Theme (eYRC 2023-24)
*                ===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ eYRC#GG#3047 ]
# Author List:		[ Names of team members worked on this file separated by Comma: Aaditya Porwal , Rudraksh Sachin Joshi ]
# Filename:			task_1a.py
# Functions:	    [`identify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(42)
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.667, gamma = 7.9):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

##############################################################

def data_preprocessing(task_1a_dataframe):

    ''' 
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that 
    there are features in the csv file whose values are textual (eg: Industry, 
    Education Level etc) These features might be required for training the model
    but can not be given directly as strings for training. Hence this function 
    should return encoded dataframe in which all the textual features are 
    numerically labeled.

    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                          Pandas dataframe read from the provided dataset 

    Returns:
    ---
    `encoded_dataframe` : [ Dataframe ]
                          Pandas dataframe that has all the features mapped to 
                          numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    #################    ADD YOUR CODE HERE    ##################
    categorical_features = ["Education", "Gender", "EverBenched", "City"]

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Perform label encoding on categorical features and replace them in the DataFrame
    for feature in categorical_features:
        task_1a_dataframe[feature] = label_encoder.fit_transform(task_1a_dataframe[feature])

    task_1a_dataframe = task_1a_dataframe.astype("int64")

    # Standardize numerical features
    num_cols = [col for col in task_1a_dataframe.columns if col != 'LeaveOrNot']
    task_1a_dataframe[num_cols] = (task_1a_dataframe[num_cols] - task_1a_dataframe[num_cols].mean()) / task_1a_dataframe[num_cols].std()
    encoded_dataframe =task_1a_dataframe
    ##############################################################

    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second 
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to 
                        numbers starting from zero

    Returns:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    #################    ADD YOUR CODE HERE    ##################
    targets = encoded_dataframe["LeaveOrNot"]
    input_features = encoded_dataframe.drop('LeaveOrNot', axis=1)
    features_and_targets = []
    features_and_targets.append(input_features)
    features_and_targets.append(targets)
    ##############################################################
    
    return features_and_targets


def load_as_tensors(features_and_targets):

    ''' 
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.
    
    Input Arguments:
    ---
    `features_and targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    #################    ADD YOUR CODE HERE    ##################
    X_train, X_test, y_train, y_test = train_test_split(features_and_targets[0], features_and_targets[1], test_size=0.3, random_state=44)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
        
    custom_dataset = CustomDataset(X_train_tensor, y_train_tensor)

    batch_size = 175
    dataloader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
    tensors_and_iterable_training_data = []
    tensors_and_iterable_training_data.append(X_train_tensor)
    tensors_and_iterable_training_data.append(X_test_tensor)
    tensors_and_iterable_training_data.append(y_train_tensor)
    tensors_and_iterable_training_data.append(y_test_tensor)
    tensors_and_iterable_training_data.append(dataloader)
    ##############################################################

    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
    '''
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers. 
    It defines the sequence of operations that transform the input data into 
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and 
    the output is the prediction of the model based on the input data.
    
    Returns:
    ---
    `predicted_output` : Predicted output for the given input data
    '''
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        '''
        Define the type and number of layers
        '''
        #######    ADD YOUR CODE HERE    #######
        self.fc1 = nn.Linear(8, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        ###################################    

    def forward(self, x):
        '''
        Define the activation functions
        '''
        #######    ADD YOUR CODE HERE    #######
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        predicted_output = x.squeeze()
        ###################################

        return predicted_output

def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    #################    ADD YOUR CODE HERE    ##################
    loss_function = FocalLoss()
    ############################################################
    
    return loss_function

def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    #################    ADD YOUR CODE HERE    ##################
    number_of_epochs = 37
    ############################################################

    return number_of_epochs

def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################    ADD YOUR CODE HERE    ##################
    optimizer = optim.Adam(model.parameters(),lr=0.004)
    ##############################################################

    return optimizer



def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''    
    #################    ADD YOUR CODE HERE    ##################
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataloader = tensors_and_iterable_training_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(number_of_epochs):
        running_loss = 0.0
        for data in dataloader:
            features, targets = data
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
    trained_model = model
    ##############################################################

    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''    
    #################    ADD YOUR CODE HERE    ##################
    X_test_tensor, y_test_tensor = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()
    
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predicted_values = trained_model(X_test_tensor)
        predicted_classes = (predicted_values >= 0.5).float()
    
    correct = (predicted_classes == y_test_tensor.view_as(predicted_classes)).sum().item()
    model_accuracy = correct / len(y_test_tensor) * 100
    ##############################################################

    return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########    
'''
    Purpose:
    ---
    The following is the main function combining all the functions
    mentioned above. Go through this function to understand the flow
    of the script

'''
if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and 
    # converting it to a pandas Dataframe
    task_1a_dataframe = pd.read_csv("D:\\HP\\users\\OneDrive\\Desktop\\eyantra\\Task_1A\\temp\\task_1a_dataset.csv")

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    
    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
                    loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "D:\\HP\\users\\OneDrive\\Desktop\\eyantra\\Task_1A\\temp\\task_1a_trained_model.pth")
