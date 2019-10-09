# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:04:32 2019

@author: Rita
"""

import numpy as np
import random

import matplotlib.pyplot as plt # pyplot are only used for ploting

###############################################################################

# load data and split into dataset and train set
# create dictionary to add attributes to the data

###############################################################################

file = np.genfromtxt('mushroom_data.txt', dtype='str')
attributes = list(["cap shape", "cap surface", "cap color", "bruises","odor",
              "gill attachment","gill spacing","gill size","gill color","stalk shape",
              "stalk root","stalk surface above ring","stalk surface below ring",
              "stalk color above ring","stalk color below ring","veil type","veil color",
              "ring number","ring type","spore print color","population","habitat","output"])


# get random index: choosing S distinct examples at random from the entire data-set


def Rand(S): 
    res = []   
    for j in range(S): 
        res.append(random.randint(0,5644))   
    return res 


def train_set(values,numbers):
    trainset = []
    for i in numbers:
        x= values[i]
        trainset.append(x)
    trainset= np.transpose(trainset)  # transform n x 23 to 23 x n
    return trainset

def testset_dict(values,sampling_id):
    testset = []
    rest_id = list(range(0,5644))
    for i in rest_id:
        if i in sampling_id:
            rest_id.remove(i)
            continue

    for i in rest_id:
        x= values[i]
        testset.append(x)
    testset= np.transpose(testset)  # transform n x 23 to 23 x n
    testset_dict=dict(zip(attributes, testset)) ##create a dictionary data structure
    return testset_dict


#############################################################################

# heuristic 1: Information thoery 
# 
# step1: find the Entropy and Information Gain
# step2: get Most Important attributes
# step3: get subtree by removing the chosen featurn
# step4: build decision tree
# step5: check the accuracy of the tree with incremental train set

# the only difference between two heuristics is the way to get most important features
# other steps are the same
    
    
#############################################################################

# Gain(A) = H(Examples) - Remainder(A)

def H_examples(trainset):
    
    output = trainset["output"]
    #Get distinct value of output "e","p", and frequncy
    unique_elements, counts = np.unique(output, return_counts=True)
    if len(counts)==1: # check purity
        p = 1
        H_examples = 0 
    else:
        p = counts / counts.sum()
        H_examples = -(p[0] * np.log2(p[0])+ p[1] * np.log2(p[1]))       
    return H_examples
 

def remainder_A(trainset,attribute):       
    
    #Get distinct feature of attributes and frequncy 
    unique_elements_in_att, counts2 = np.unique(trainset[attribute], return_counts=True)
    fraction1 = counts2/len(trainset["output"])
    counter = 0
    entropy = []    
    for feature in unique_elements_in_att:
        indices = np.where(trainset[attribute]==feature) 
        output = trainset["output"][indices]
        entropy_per_feature = 0
        feature_count = counts2[counter]       
        #Get distinct value("e","p") of each feature 
        unique_feature, counts_feature = np.unique(output, return_counts=True) 
        for feature_label_count in counts_feature:
            fraction2 = feature_label_count/feature_count 
            #Get entropy of each feature 
            entropy_per_feature += -fraction2*np.log2(fraction2)
        entropy.append(entropy_per_feature)
        counter += 1
    res = np.dot(fraction1,entropy)
    return res       

    
def most_important(trainset):    
       
    # calculate Information Gain for every attributes    
    ig = []   
    for i in attributes[:-1]:
        ig.append(H_examples(trainset)-remainder_A(trainset,i))  # Gain(A) = H(Examples) - Remainder(A)      
    maxpos= ig.index(max(ig))   
    return attributes[maxpos]

# remove the chosen feature, we divide the data according to the values of this feature
# and recursively build subtrees out of each partial data-set

def get_subtree(trainset, parent_guess, value):
    indices = np.where(trainset[parent_guess]==value)     
    subtree = []
    for keys in trainset:
        subtree.append(trainset[keys][indices])
    sub_dict = dict(zip(attributes, subtree))
    return sub_dict


def decision_tree(trainset,tree=None): 

    #Get attribute with maximum information gain
    parent_guess = most_important(trainset)
    values = np.unique(trainset[parent_guess])
    
    if tree is None:                    
        tree={}
        tree[parent_guess] = {}

    for value in values:
        subtree = get_subtree(trainset,parent_guess,value)
        Value,counts = np.unique(subtree["output"],return_counts=True)  
                                       
        #Checking purity, and stop when data is pure    
        if len(counts)==1:
            tree[parent_guess][value] = Value[0]                                                    
        else:        
            tree[parent_guess][value] = decision_tree(subtree)  
           
    return tree

##############################################################################
# step5: check the accuracy of the tree with incremental train set 

def predict(sample,tree):
    #This function is used to predict for any input variable     
    #Recursively we go through the tree that we built earlier
    for nodes in tree.keys():  
        value = sample[nodes]
        if value not in tree[nodes]:
            prediction = "p"
        else:
            tree = tree[nodes][value]
            prediction = 0
            if type(tree) is dict:
                prediction = predict(sample, tree)
            else:
                prediction = tree
                break;                            
    return prediction


def test_id(sampling_id):
    rest_id = list(range(0,5644))
    for i in rest_id:
        if i in sampling_id:
            rest_id.remove(i)
    return rest_id



def success_rate(testset_dict,test_id,sampling_id,increment_id): 
    accuracy_collection = []
    numbers = 0
    # get dataset with increment and build the tree
    for i in increment_id:
        print('Running with {} examples in training set'.format(i))
        numbers = sampling_id[:i]
        trainset= train_set(file,numbers)
        trainset_dict = dict(zip(attributes, trainset))
        tree = decision_tree(trainset_dict,tree=None)   

        #get test samples to predict
        label = []  
        for n in test_id:
            sample = dict(zip(attributes, file[n]))
            a= predict(sample,tree)
            label.append(a)
        success = label == testset_dict["output"] 
        accuracy = success.mean()
        print('Given current tree, we have a success rate of {} percent'.format(accuracy))  
        accuracy_collection.append(accuracy)
    return accuracy_collection

##############################################################################

### Main__Function

    
S =500# please enter a training set size (a positive multiple of 250 that is <= 1000): 
I = 25  #Please enter a training increment (either 10, 25, or 50): 
sampling_id = Rand(S)    
increment_id = list(range(I,S+I,I))
test_id = test_id(sampling_id)
testset_dict = testset_dict(file,sampling_id)
accuracy_collection = success_rate(testset_dict,test_id,sampling_id,increment_id) 

##### get final tree
trainset= train_set(file,sampling_id)
trainset_dict = dict(zip(attributes, trainset))
final_tree = decision_tree(trainset_dict,tree=None)  
print("final_tree",final_tree)

# plot
x = [0]
y = [float(0)]
x.extend(increment_id)
y.extend(accuracy_collection)
plt.plot(x, y)
plt.xlabel("size of train set")
plt.ylabel("correction")
