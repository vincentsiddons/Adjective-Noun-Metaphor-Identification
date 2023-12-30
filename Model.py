import os
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pyplot as plt
from Preprocessing import Preprocessing

class Model:
    training = 0.0
    valid = 0.0
    test = 0.0
    
    curr_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, training = 0.8, valid = 0.1, test = 0.1):
        self.training = training
        self.valid = valid
        self.test = test
    
    @classmethod
    def no_arg(cls):
        return cls()
    
    #Get the indicies of where two words don't match (2d array is sorted alphabetically)
    def get_indicies(self, twoD_arr):
        ind_arr = []
        for j in range(0, len(twoD_arr) - 1):
            if twoD_arr[j][0] != twoD_arr[j + 1][0]:
                ind_arr.append(j)

        return ind_arr
    
    #Calculates the F1 score given values in a 2d array of 1 by 2 matricies, with the first element being the predicted value and the second being the true value
    def calculate_F1_score(self, predictions):
        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0

        for j in range(0, len(predictions)):
            if predictions[j][0] < 0.5 and predictions[j][1] == 0:
                t_n += 1
            elif predictions[j][0] > 0.5 and predictions[j][1] == 0:
                f_p += 1
            elif predictions[j][0] < 0.5 and predictions[j][1] == 1:
                f_n += 1
            elif predictions[j][0] > 0.5 and predictions[j][1] == 1:
                t_p += 1
        precision = t_p/(t_p + f_p)
        recall = t_p/(t_p + f_n)
        f1 = (2*precision*recall)/(precision + recall)
        
        return f1

    def create_ROC(self, predictions):
        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0
        recalls = []
        fprs = []

        for j in range(0, len(predictions)):
            if predictions[j][0] < 0.5 and predictions[j][1] == 0:
                t_n += 1
            elif predictions[j][0] > 0.5 and predictions[j][1] == 0:
                f_p += 1
            elif predictions[j][0] < 0.5 and predictions[j][1] == 1:
                f_n += 1
            elif predictions[j][0] > 0.5 and predictions[j][1] == 1:
                t_p += 1
            try:
                recalls.append(t_p/(t_p + f_n))
                fprs.append(f_p/(f_p + t_n))
            except:
                continue

        plt.plot(fprs, recalls)
        plt.show()
     

    #Given input and output examples, learn from them and give weights to be used for testing and validation
    def train(self, learning_rate, relu_negative_slope, epochs):

        words = Preprocessing.no_arg()
        word_info = words.get_word2vec_embeddings()

        #This is a numpy array of the adjective, the noun, its classification, and (if found) its embedding
        word_info = self.get_word2vec_embeddings()

        indicies = self.get_indicies(word_info)

        #Train to gain weights on each training set for each word by using training indicies from 0 -> 1 and then from 2 -> 3 ...
        w = t.randn((300, 600), requires_grad = True, generator = t.manual_seed(42))
        b = t.randn((300,1), requires_grad = True, generator = t.manual_seed(43))

        #Converts weights and biases to output
        u = t.randn((300,1), requires_grad = True, generator = t.manual_seed(44))

        #Performs gradient descent on weights
        optimizer = t.optim.Adam((w, b, u), learning_rate)

        for k in range(0, epochs):
            for j in range(0, len(indicies) - 1):
                for i in range(indicies[j], indicies[j] + (int(self.training*(indicies[j + 1] - indicies[j])))):
                    #If the adj-noun pair doesn't have an embedding, then just move on
                    try:
                        x = t.tensor(word_info[i][3], dtype=t.float)
                        x.requires_grad_(True)
                    except:
                        continue

                    #leaky RELU due to RELU giving sparce and zero gradients
                    h = t.nn.functional.leaky_relu(w @ x + b, negative_slope = relu_negative_slope)
                    h = t.transpose(h, 0, 1)
                    z = t.nn.functional.leaky_relu(h @ u, negative_slope = relu_negative_slope)

                    #The correct classification (either 1 or 0)
                    golden = np.array(word_info[i][2])
                    golden = t.tensor(golden[np.newaxis, :], dtype=t.float)
                    golden.requires_grad_(True)

                    #Calculating the cross entropy loss between z (logits) and the true value
                    loss = t.nn.functional.binary_cross_entropy_with_logits(z, golden)

                    #backpropigation
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step() 

        return w, b, u 
    
    def validate(self, learning_rate, relu_negative_slope, ephochs):

        #This is a numpy array of the adjective, the noun, its classification, and (if found) its embedding
        word_info = self.get_word2vec_embeddings()

        indicies = self.get_indicies(word_info)

        w, b, u = self.train(learning_rate, relu_negative_slope, ephochs)

        predictions = []
        
        for j in range(0, len(indicies) - 1):
            for i in range(indicies[j] + int(self.training*(indicies[j + 1] - indicies[j])), (indicies[j] + int(self.training*(indicies[j + 1] - indicies[j])) + int(self.valid*(indicies[j + 1] - indicies[j])))):
                try:
                    x = t.tensor(word_info[i][3], dtype=t.float)
                    x.requires_grad_(True)
                except:
                    continue

                #leaky RELU due to RELU giving sparce and zero gradients
                h = t.nn.functional.leaky_relu(w @ x + b, negative_slope = relu_negative_slope)
                h = t.transpose(h, 0, 1)
                z = t.nn.functional.leaky_relu(h @ u, negative_slope = relu_negative_slope)
                y = t.nn.functional.sigmoid(z)
                y = y[0][0]
                y = y.item()
                #The correct classification (either 1 or 0)
                golden = word_info[i][2][0]
                predictions.append([y, golden]) 
        f1 = self.calculate_F1_score(predictions)

        return f1

    def testing(self, learning_rate, relu_negative_slope, epochs):
         #This is a numpy array of the adjective, the noun, its classification, and (if found) its embedding
        word_info = self.get_word2vec_embeddings()

        indicies = self.get_indicies(word_info)

        w, b, u = self.train(learning_rate, relu_negative_slope, epochs)

        predictions = []
        
        for j in range(0, len(indicies) - 1):
            for i in range(indicies[j] + int(self.training*(indicies[j + 1] - indicies[j])) + int(self.valid*(indicies[j + 1] - indicies[j])), (indicies[j + 1])):
                try:
                    x = t.tensor(word_info[i][3], dtype=t.float)
                    x.requires_grad_(True)
                except:
                    continue
                #leaky RELU due to RELU giving sparce and zero gradients
                h = t.nn.functional.leaky_relu(w @ x + b, negative_slope = relu_negative_slope)
                h = t.transpose(h, 0, 1)
                z = t.nn.functional.leaky_relu(h @ u, negative_slope = relu_negative_slope)
                y = t.nn.functional.sigmoid(z)
                y = y[0][0]
                y = y.item()
                #The correct classification (either 1 or 0)
                golden = word_info[i][2][0]
                predictions.append([y, golden])  
        f1 = self.calculate_F1_score(predictions)

        return f1




    
    