# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:47:22 2019

Introduction to Machine Learning
Assignment 01, Part 01

@author: pengyuanchen
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


###############################################################################

# heuristic 2: Base Counting
# 
# step1: get score of each attributes by comparing how many examples I would classify correctly
# step2: get Most Important attributes
# step3: get subtree by removing the chosen feature
# step4: build decision tree
# step5: check the accuracy of the tree with incremental train set

# the only difference between two heuristics is the way to get most important features
# other steps are the same
    
###############################################################################


def get_score(trainset,attribute):
    unique_elements_in_att, counts1 = np.unique(trainset[attribute], return_counts=True)
    ttl_score= 0
    score_per_freature = 0
    for feature in unique_elements_in_att:
        indices = np.where(trainset[attribute]==feature)
        output = trainset["output"][indices]
        _, counts2 = np.unique(output, return_counts=True)
        if len(counts2) ==1:
            score_per_freature = 1
        elif counts2[0] >= counts2[1]:
            score_per_freature = counts2[0]/(counts2[0]+counts2[1])
        else:
            score_per_freature = counts2[1]/(counts2[0]+counts2[1])
    ttl_score +=score_per_freature
    return ttl_score


        
def most_important(trainset):
    score = []
    for i in attributes[:-1]:
        score.append(get_score(trainset,i)) 
    maxpos= score.index(max(score))
    return attributes[maxpos]



def get_subtree(trainset, parent_guess, value):
    indices = np.where(trainset[parent_guess]==value)     
    subtree = []
    for keys in trainset:
        subtree.append(trainset[keys][indices])
    sub_dict = dict(zip(attributes, subtree))
    return sub_dict


def decision_tree(trainset,tree=None): 

    parent_guess = most_important(trainset) 
    values = np.unique(trainset[parent_guess])

    if tree is None:                    
        tree={}
        tree[parent_guess] = {}

    for value in values:
    
        subtree = get_subtree(trainset,parent_guess,value)
        Value,counts = np.unique(subtree["output"],return_counts=True)  

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

## plot

x = [0]
y = [float(0)]
x.extend(increment_id)
y.extend(accuracy_collection)
plt.plot(x, y)
plt.xlabel("size of train set")
plt.ylabel("correction")
