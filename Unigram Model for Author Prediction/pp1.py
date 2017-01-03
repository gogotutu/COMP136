import math
import requests
import numpy as np

# Read in files
train = requests.get("http://www.cs.tufts.edu/comp/136/HW/pp1files/training_data.txt").text
test = requests.get("http://www.cs.tufts.edu/comp/136/HW/pp1files/test_data.txt").text

# Split the data into a list of words
train_data = train.split()
test_data = test.split()

# Union the words in both the training set and the test set to build a dictionary
vocabulary1 = set(train_data)|set(test_data)
K = len(vocabulary1)

# Break up into 5 training sets of different sizes

N = len(train_data)
N1 = N/128
N2 = N/64
N3 = N/16
N4 = N/4
N5 = N

train_data_1 = train_data[0:N1]
train_data_2 = train_data[0:N2]
train_data_3 = train_data[0:N3]
train_data_4 = train_data[0:N4]
train_data_5 = train_data[0:N5]

# This function compute the Perplexity on the test set
# It takes in three arguments: test data, training data, and a vocabulary of K words
def compute_PP(test_data, train_data, vocabulary):

    N_train = len(train_data)
    N_test = len(test_data)
    K = len(vocabulary)
    
    # Learn the model using trainning data: 
    # For each word in dictionary, count the number of times it appears in the training set
    dictionary = dict.fromkeys(vocabulary,0)
    for word in train_data:
        dictionary[word] = dictionary[word] + 1
    
    
    # Compute the perplexity for each method 
    ml, ma, pd = 0, 0, 0
    for word in test_data:
        ml = ml + math.log(float(max(0.1, dictionary[word]))/N_train)
        ma = ma + math.log(float(dictionary[word] + 2 - 1)/(N_train + 2*K - K))
        pd = pd + math.log(float(dictionary[word] + 2)/(N_train + 2*K))
    
    PP_ML = math.exp(-1./N_test * ml)
    PP_MA = math.exp(-1./N_test * ma)
    PP_PD = math.exp(-1./N_test * pd)
    
    return [PP_ML, PP_MA, PP_PD]

print "************************************************************"
print "***                                                      ***"
print "***                       Task 1                         ***"
print "***                                                      ***"
print "************************************************************"

# Train the data using 5 different training sets, and test on the test sets
row_1_test = compute_PP(test_data,train_data_1,vocabulary1)
row_2_test = compute_PP(test_data,train_data_2,vocabulary1)
row_3_test = compute_PP(test_data,train_data_3,vocabulary1)
row_4_test = compute_PP(test_data,train_data_4,vocabulary1)
row_5_test = compute_PP(test_data,train_data_5,vocabulary1)

print "Perplexity on test sets, with ML, MAP, Pred. Dist."
print row_1_test
print row_2_test
print row_3_test
print row_4_test
print row_5_test

# We repeat the above, but now test on training set itself
row_1_train = compute_PP(train_data,train_data_1,vocabulary1)
row_2_train = compute_PP(train_data,train_data_2,vocabulary1)
row_3_train = compute_PP(train_data,train_data_3,vocabulary1)
row_4_train = compute_PP(train_data,train_data_4,vocabulary1)
row_5_train = compute_PP(train_data,train_data_5,vocabulary1)

print "Perplexity on training sets, with ML, MAP, Pred. Dist."
print row_1_test
print row_2_test
print row_3_test
print row_4_test
print row_5_test


print "************************************************************"
print "***                                                      ***"
print "***                       Task 2                         ***"
print "***                                                      ***"
print "************************************************************"

# Initialization
alpha_prime = np.arange(1,11)
log_evidence_vec = []
test_perplexity_vec = []

# Use the training set of size N/128 and train the model
dictionary1 = dict.fromkeys(vocabulary1,0)
for word in train_data_1:
    dictionary1[word] = dictionary1[word] + 1

# Loop over different values of alpha and compute both log evidence and perplexity
for alpha in alpha_prime:
    
    alpha_0 = alpha * K
    
    # Recursively compute the log evidence function, avoiding compute the factorial
    le = 0
    for word in dictionary1:
        for k in np.arange(0, dictionary1[word]):
            le = le + math.log(alpha + k)
    for k in np.arange(0, N1):
        le = le - math.log(alpha_0 + k)    
    log_evidence_vec.append(le)
    
    # Recursively compute the perplexity
    pd = 0
    for word in test_data:
        pd = pd + math.log(float(dictionary1[word] + alpha)/(N1 + alpha * K))
    test_perplexity = math.exp(-1./N * pd)
    test_perplexity_vec.append(test_perplexity)


print "log_evidence = ", log_evidence_vec
print "test_perplexity = ", test_perplexity_vec


print "************************************************************"
print "***                                                      ***"
print "***                       Task 3                         ***"
print "***                                                      ***"
print "************************************************************"



text_1 = requests.get("http://www.cs.tufts.edu/comp/136/HW/pp1files/pg84.txt.clean").text
text_2 = requests.get("http://www.cs.tufts.edu/comp/136/HW/pp1files/pg345.txt.clean").text
text_3 = requests.get("http://www.cs.tufts.edu/comp/136/HW/pp1files/pg1188.txt.clean").text

pg84 = text_1.split()
pg345 = text_2.split()
pg1188 = text_3.split()

# Build a new vocabulary for task 3
vocabulary2 = set(pg84) | set(pg345) | set(pg1188)
len(vocabulary2)

r1 = compute_PP(pg84, pg345, vocabulary2)
r2 = compute_PP(pg1188, pg345, vocabulary2)

print "Perplexity on pg84 = ", r1[2]
print "Perplexity on pg1188 = ", r2[2]