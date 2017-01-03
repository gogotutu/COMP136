
# coding: utf-8

# In[1]:

import random
import sys
import pandas as pd
import numpy as np


# ## Task 1: Regularization

# In[2]:

# Create three additional training sets
path = 'data/train-1000-100.csv'
df = pd.read_csv(path, header = None)
df[0: 50].to_csv('data/train-50(1000)-100.csv',header = None, index = None)
df[0:100].to_csv('data/train-100(1000)-100.csv',header = None, index = None)
df[0:150].to_csv('data/train-150(1000)-100.csv',header = None, index = None)

path2 = 'data/trainR-1000-100.csv'
df2 = pd.read_csv(path2, header = None)
df2[0: 50].to_csv('data/trainR-50(1000)-100.csv',header = None, index = None)
df2[0:100].to_csv('data/trainR-100(1000)-100.csv',header = None, index = None)
df2[0:150].to_csv('data/trainR-150(1000)-100.csv',header = None, index = None)


# In[3]:

def reg_linreg(filename):
    
    # Handle the case for the three data sets created, the filenames for the training data
    # and test data are different
    if '(' in filename:
        train_filename = filename
        test_filename = filename[filename.find('(')+1:filename.find(')')] + filename[filename.find(')')+1:]
    else:
        train_filename = filename
        test_filename = filename
    
    # Read in corresponding files
    train_file  = 'data/train-' + train_filename + '.csv'
    train_label = 'data/trainR-' + train_filename + '.csv'
    test_file = 'data/test-' + test_filename + '.csv'
    test_label = 'data/testR-' + test_filename + '.csv'
    
    phi_train = pd.read_csv(train_file, header = None).values
    t_train = pd.read_csv(train_label, header = None).values
    phi_test = pd.read_csv(test_file, header = None).values
    t_test = pd.read_csv(test_label, header = None).values
    
    lam_vec = np.arange(0,150)
    MSE_train_vec = []
    MSE_test_vec = []
    
    for lam in lam_vec:
        diag = np.ones(phi_train.shape[1])*lam
        LAM = np.diag(diag)
        w = np.dot(np.dot(np.linalg.inv(LAM + np.dot(phi_train.transpose(),phi_train)),phi_train.transpose()),t_train) # Compute w with respect to equation (3.28) in Bishop
        MSE_train = ((np.dot(phi_train,w) - t_train)**2).sum()/float(t_train.shape[0]) # Compute the MSE for trainning set
        MSE_train_vec.append(MSE_train)
        MSE_test = ((np.dot(phi_test,w) - t_test)**2).sum()/float(t_test.shape[0]) # Compute the MSE for test set
        MSE_test_vec.append(MSE_test)
    
    lam_opt = lam_vec[np.argmin(MSE_test_vec)] # The optimal lambda, when MSE reaches minimum
    MSE_opt = min(MSE_test_vec)
    
    return MSE_train_vec, MSE_test_vec, lam_opt, MSE_opt


# In[4]:

lam_vec = np.arange(0,150)


# In[5]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('100-10')
MSE_true = [3.78] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[6]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('100-100')
MSE_true = [3.78] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[7]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('50(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[8]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('100(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[9]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('150(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[10]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('1000-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[11]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('crime')
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# In[12]:

MSE_train, MSE_test, lam_opt, MSE_opt = reg_linreg('wine')
print 'The optimal lambda is: ', lam_opt
print 'The best test-set MSE is: ', MSE_opt


# ## Discussion:
# 
# * **Q: Why can't the trainning set be used to select $\lambda$?**
# 
#     **A: ** From the above plots, we can see that the MSE on the training set is minimized at $\lambda = 0$ and is monotonically increasing with the increase of $\lambda$.
#     
#     This is not surprising because we know that when $\lambda = 0$, $w$ correponds to the ordinary least square: $$w = (\Phi^T \Phi)^{-1} \Phi^T t$$ and we know that the ordinary least square method would give a result that minimizes the Sum of Squared Residuals (SSR): $$ SSR = \sum_{i = 1}^N  \left( \phi(x_i)^T w - t_i \right)^2$$ on the training set. That means it would also minimize the MSE, which is $1/N$ times the SSR, on the training set.
#     
#     
# * **Q: How does $\lambda$ affect error on the test set?**
# 
#     **A: ** For all examples above, increasing $\lambda$ would firstly decrease and then increase the MSE on test set. When $\lambda = 0$, the MSE on the training set is minimized, this is when we say the model is "overfitting", it fits the training set so well that cannot be generalized to the test set. When $\lambda$ is very large, then the regularization dominates $w$ and therefore introducing too much bias, and that is when we say the model is "underfitting".
#     
#     Therefore, there exists an optimal choice of $\lambda$, neither too little nor too much regulariztion is good. However, in practice, we cannot use test set doing this. We can use either cross validation or Bayesian linear regression as later in this assignment.
#     
#     
# * **Q: How does the choice of the optimal $\lambda$ vary with the number of features and the number of examples? Consider both the cases where the number of features is fixed and where the number of examples is fixed. How do you explain these variations?**
# 
#     **A: ** For fixed number of features, like in the sets "50(1000)-100", "100(1000)-100", "150(1000)-100" and "1000-100", the optimal $\lambda$ are 8, 19, 23, 27 respectively, which is monotonically increasing with the number of examples. 
#     
#     For fixed number of examples, like in the sets "100-10" and "100-100", the optimal $\lambda$ are 8 and 22 respective, which is also increasing with the number of features.
#     
#     To understand this, think about when we are doing regularized linear regression we are actually minimizing the following quantity: $$ \sum_{i = 1}^N  \left( \phi(x_i)^T w - t_i \right)^2 + \lambda \sum_{j=1}^M w_j^2$$ $N$ is the number of examples, and $M$ is the number of features.
#     
#     When $M$ is fixed and $N$ increasing, the above quantity is more and more dominated by the Sum of Squared Residual (SSR) term, the solution would converge to the ordinary least square and therefore becomes overfitting. This is balanced by using larger $\lambda$. It would put more weights on the regularization term and avoid overfitting.
#     
#     When $N$ is fixed and $M$ increasing, fitting to more features means looking for models with more variance, which is also at the risk of overfitting. Therefore we also need larger $\lambda$ to avoid overfitting.

# ## Task 2: Learning Curves

# In[13]:

def learning_curve(lam):
    
    # Read in corresponding files and store them as numpy arrays
    train_file  = 'data/train-1000-100.csv'
    train_label = 'data/trainR-1000-100.csv'
    test_file = 'data/test-1000-100.csv'
    test_label = 'data/testR-1000-100.csv'
    
    phi_train = pd.read_csv(train_file, header = None).values
    t_train = pd.read_csv(train_label, header = None).values
    phi_test = pd.read_csv(test_file, header = None).values
    t_test = pd.read_csv(test_label, header = None).values
    
    size_vec = range(10,810,20) # Sample sizes range from 10 to 800 with a step size of 20
    MSE_vec = []
    for size in size_vec:
        MSE_ave = []
        for k in range(20): # For each sample size, repeat 20 times to compute the average MSE
            rows = [random.randint(0,999) for i in range(size)] # Randomly pick rows from the sample
            phi = phi_train[rows]
            t = t_train[rows]
            diag = np.ones(phi.shape[1])*lam
            LAM = np.diag(diag)
            w = np.dot(np.dot(np.linalg.inv(LAM + np.dot(phi.transpose(),phi)),phi.transpose()),t)
            MSE = ((np.dot(phi_test,w) - t_test)**2).sum()/float(t_test.shape[0]) # Compute the MSE for test set
            MSE_ave.append(MSE)
        MSE_vec.append(np.mean(MSE_ave))
    return MSE_vec


# In[14]:

size_vec = range(10,810,20)
MSE_true = [4.015] * len(size_vec)
lc_10 = learning_curve(10)
lc_27 = learning_curve(27)
lc_100 = learning_curve(100)


# In[15]:
print lc_10
print lc_27
print lc_100


# ## Discussion:
# 
# * **Q: What can you observe from the plots regarding the dependence on $\lambda$ and the number of samples?**
#     
#     **A: ** For three *representative* values of $\lambda$, $\lambda = 8$ overfitting, $\lambda = 27$ optimal, and $\lambda = 100$ underfitting, the test set MSE would decrease with the increase in the number of samples. This is because, for a fixed $\lambda$, more data points will reduce the variance and therefore reduce the MSE.
#     
#     
# * **Q: Consider both the case of small training set sizes and large training set sizes. How do you explain these variations?**
# 
#     When the data size is small (less than 100 points), the relationship is $$ MSE_{100} > MSE_{27} > MSE_{10} $$ When the size of the trainning set is small, the bias introduced by $\lambda$ is large and dominating, and therefore the larger $\lambda$ the larger test set MSE.
#     
#     When the data size is large (near 800), the relationship is $$MSE_{10} > MSE_{100} > MSE_{27} $$, since we know that $\lambda  = 27$ is the optimal choice of $\lambda$ for the "1000-100" data. $MSE_{27}$ is smaller than the other two.

# ## Task 3: Cross Validation

# In[17]:

def cross_val(filename):
    
    
    if '(' in filename:
        train_filename = filename
        test_filename = filename[filename.find('(')+1:filename.find(')')] + filename[filename.find(')')+1:]
    else:
        train_filename = filename
        test_filename = filename
    
    # Read in corresponding files and store them as numpy arrays
    train_file  = 'data/train-' + train_filename + '.csv'
    train_label = 'data/trainR-' + train_filename + '.csv'
    test_file = 'data/test-' + test_filename + '.csv'
    test_label = 'data/testR-' + test_filename + '.csv'
    
    phi_train = pd.read_csv(train_file, header = None).values
    t_train = pd.read_csv(train_label, header = None).values
    phi_test = pd.read_csv(test_file, header = None).values
    t_test = pd.read_csv(test_label, header = None).values

    n = phi_train.shape[0]
    d = n/10
    
    SEED = 7485
    ind = range(0,n)
    random.seed(SEED)
    random.shuffle(ind) #Randomly shuffle the indices
    
    lam_vec = np.arange(0,150)
    MSE_vec = []

    for lam in lam_vec:
        MSE_cv = []
        for k in range(0,10): # 10 fold validation
            
            ind_test = ind[k*d : (k+1)*d] # d = n/10, split the data into test sets and validation sets
            ind_train = ind[0:k*d] + ind[(k+1)*d:]
            
            phi = phi_train[ind_train]
            t = t_train[ind_train]
            phi_t = phi_train[ind_test]
            t_t = t_train[ind_test]

            diag = np.ones(phi_train.shape[1])*lam
            LAM = np.diag(diag) 
            w = np.dot(np.dot(np.linalg.inv(LAM + np.dot(phi.transpose(),phi)),phi.transpose()),t) # Compute w with respect to equation (3.28) in Bishop
            MSE = ((np.dot(phi_t,w) - t_t)**2).sum()/float(t_t.shape[0])
            MSE_cv.append(MSE)
        MSE_vec.append(np.mean(MSE_cv))
    
    # Choose the optimal lambda and train the model using the entire trainning set
    lam_opt = lam_vec[np.argmin(MSE_vec)] # Choose the optimal lambda which minimize the average MSE on cross validation sets
    diag = np.ones(phi_train.shape[1])*lam_opt
    LAM = np.diag(diag) 
    w = np.dot(np.dot(np.linalg.inv(LAM + np.dot(phi_train.transpose(),phi_train)),phi_train.transpose()),t_train) # Compute w with respect to equation (3.28) in Bishop
    MSE_test = ((np.dot(phi_test,w) - t_test)**2).sum()/float(t_test.shape[0]) # Compute the MSE for test set

    return MSE_vec, lam_opt, MSE_test


# In[18]:

MSE_vec, lam_opt, MSE_test = cross_val('100-10')
MSE_true = [3.78] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[19]:

MSE_vec, lam_opt, MSE_test = cross_val('100-100')
MSE_true = [3.78] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[20]:

MSE_vec, lam_opt, MSE_test = cross_val('50(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[21]:

MSE_vec, lam_opt, MSE_test = cross_val('100(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[22]:

MSE_vec, lam_opt, MSE_test = cross_val('150(1000)-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[23]:

MSE_vec, lam_opt, MSE_test = cross_val('1000-100')
MSE_true = [4.015] * len(lam_vec)
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[24]:

MSE_vec, lam_opt, MSE_test = cross_val('crime')
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# In[25]:

MSE_vec, lam_opt, MSE_test = cross_val('wine')
print 'The optimal lambda is: ', lam_opt
print 'The test set MSE is: ', MSE_test


# ## Discussion:
# 
# * **Q: How do the results compare to the best test-set results from part 1 both in terms of the choice of $\lambda$ and test set MSE?**
# 
#     **A: ** The cross validation tends to give larger choice of the optimal $\lambda$ compared to the results from part 1 for most of the cases. However, in most of the casese, the optimal $lambda$ chosen is close to that from part 1. There are cases that the difference is large, but those are the cases when the MSE curve is very flat, and therefore the difference in MSE is not large.
#     
#     The test MSE is strictly larger than the results from part 1, this is because from part 1, $\lambda$ were chosen by minimizing test set MSE. By using only cross validation, it is very unlikely we can choose the same $\lambda$ that can minimize the test set MSE. As long as we chose a different $\lambda$, the test set MSE is sure to be larger.
#     
#     
# * **Q: What is the run time cost of this scheme?**
# 
#     **A: ** The run time cost of this scheme is high. This is because for each value of $\lambda$, we need to run 10-fold cross validation and compute the average validation MSE. We need to train the model 10 times on 9 out of 10 cross validation sets and test 10 time on the one left cross validation set correspondingly.
#     
#     
# * **Q: How does the quality depend on the number of examples and features?**
# 
#     **A: ** The quality is better when the number of examples is much larger than the number of features, like in the "1000-100" case. When the number of examples is close to or even less than the number of features. The matrix $\Phi^T \Phi$ could non-invertable. Therefore, a lot of regularization is needed and therefore introducing a lot of bias.

# ## Task 4: Bayesian Linear Regression

# In[26]:

def model_selection(filename):
    
    # Read in corresponding files and store them as numpy arrays
    train_file  = 'data/train-' + filename + '.csv'
    train_label = 'data/trainR-' + filename + '.csv'
    
    phi = pd.read_csv(train_file, header = None).values
    t = pd.read_csv(train_label, header = None).values
    
    M = phi.shape[1]
    N = phi.shape[0]
    
    alpha = 1
    beta = 1
    tol = 1
    while tol > 0.0001:
        SN = np.linalg.inv(alpha * np.identity(M) + beta * np.dot(phi.transpose(),phi))
        mN = beta * np.dot(SN, np.dot(phi.transpose(),t))
        
        LAM = np.linalg.eigvals(beta * np.dot(phi.transpose(), phi))
        gamma = sum(LAM / (alpha + LAM))
        alpha_new = gamma / (mN * mN).sum()
        
        SIGMA = 0
        for n in range(N):
            SIGMA += (t[n][0] - (mN.transpose() * phi[n]).sum()) ** 2
        beta_new = (N - gamma) / SIGMA
        
        tol = abs(alpha_new - alpha)/alpha + abs(beta_new - beta)/alpha
        alpha = alpha_new
        beta = beta_new
    
    return alpha, beta


# In[27]:

def bayesian_regression(filename):
    
    alpha, beta = model_selection(filename)
    lam = (alpha / beta).real
    
    # For the three data sets created by myself, the filenames for the training data
    # and the test data are different.
    if '(' in filename:
        train_filename = filename
        test_filename = filename[filename.find('(')+1:filename.find(')')] + filename[filename.find(')')+1:]
    else:
        train_filename = filename
        test_filename = filename
    
    # Read in corresponding files and store them as numpy arrays
    train_file  = 'data/train-' + train_filename + '.csv'
    train_label = 'data/trainR-' + train_filename + '.csv'
    test_file = 'data/test-' + test_filename + '.csv'
    test_label = 'data/testR-' + test_filename + '.csv'
    
    phi_train = pd.read_csv(train_file, header = None).values
    t_train = pd.read_csv(train_label, header = None).values
    phi_test = pd.read_csv(test_file, header = None).values
    t_test = pd.read_csv(test_label, header = None).values
    
    diag = np.ones(phi_train.shape[1])*lam
    LAM = np.diag(diag) # Diagonal matrix with value lambdas
    w = np.dot(np.dot(np.linalg.inv(LAM + np.dot(phi_train.transpose(),phi_train)),phi_train.transpose()),t_train) # Compute w with respect to equation (3.28) in Bishop
    MSE_test = ((np.dot(phi_test,w) - t_test)**2).sum()/float(t_test.shape[0]) # Compute the MSE for test set
    
    return lam, MSE_test


# In[28]:

print 'Bayesian Regression Model Selection:'
print
lam, MSE = bayesian_regression('100-10')
print '100-10'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('100-100')
print '100-100'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('50(1000)-100')
print '50(1000)-100'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('100(1000)-100')
print '100(1000)-100'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('150(1000)-100')
print '150(1000)-100'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('1000-100')
print '1000-100'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('crime')
print 'crime'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print
lam, MSE = bayesian_regression('wine')
print 'wine'
print 'The optimal lambda is: ', lam
print 'The test set MSE is: ',  MSE
print


# ## Discussion:
# 
# * **Q: How do the results compare to the best test-set results from part 1 both in terms of the choice of $\lambda$ and test set MSE?**
# 
#     **A: ** The Bayesian linear regression tends to give smaller choice of the optimal $\lambda$ compared to the results from part 1 for most of the cases. In most of the casese, the optimal $lambda$ chosen is close to that from part 1 and the test MSE is close but strictly larger than that from part 1 for the same reason.
#     
#     
# * **Q: What is the run time cost of this scheme?**
# 
#     **A: ** The run time cost of this scheme is much lower than the case using cross validation. For each trainning data set, we can choose our $\lambda$ directly by iteratively compute $\alpha$ and $\beta$. The convergence of $\alpha$ and $\beta$ are fast, and we don't need to do the splitting, training and testing as before. Therefore, the run time cost is largely conserved.
#     
#     
# * **Q: How does the quality depend on the number of examples and features?**
# 
#     **A: ** For the same reason, the quality is better when the number of examples is much larger than the number of features, like in the "1000-100" case. When the number of examples is close to or even less than the number of features. The matrix $\Phi^T \Phi$ could non-invertable. Therefore, a lot of regularization is needed and therefore introducing a lot of bias.
# 

# ## Task 5: Comparison

# In[29]:

files = ['100-10','100-100','50(1000)-100','100(1000)-100','150(1000)-100','1000-100','crime','wine']
lam_reg_vec = []
lam_cro_vec = []
lam_bay_vec = []
MSE_reg_vec = []
MSE_cro_vec = []
MSE_bay_vec = []
for name in files:
    _,_,lam_reg,MSE_reg = reg_linreg(name)
    _,lam_cro, MSE_cro = cross_val(name)
    lam_bay, MSE_bay = bayesian_regression(name)
    
    lam_reg_vec.append(lam_reg)
    lam_cro_vec.append(lam_cro)
    lam_bay_vec.append(lam_bay)
    MSE_reg_vec.append(MSE_reg)
    MSE_cro_vec.append(MSE_cro)
    MSE_bay_vec.append(MSE_bay)


# In[35]:

print lam_reg_vec
print lam_cro_vec
print lam_bay_vec

# In[38]:

print MSE_reg_vec
print MSE_cro_vec
print MSE_bay_vec


# ## Discussion:
# 
# * **Q: How do the two model selection methods compare in terms of the test set MSE and in terms of run time?**
# 
#     **A: ** Cross validation tends to give slightly better test set MSE while Bayesian linear regression has a large advatange in saving run time.
# 
# 
# * **Q: What are the important factors affecting performance for each method?**
# 
#     **A: ** 
#     * Cross Validation Pros and Cons:
#         * Pros: Easy to understand and implement; Tends to generalize better on the test set (lower test MSE); No need for assumptions on the model
#         * Cons: High run time cost; Randomness introduced when spliting; Need to pre-set a range for $\lambda$; Cannot avoid correlations between cross validation sets
#         
#     * Bayesian Model Selection Pros and Cons:
#         * Pros: Low run time cost; Converge to $\lambda$ automatically (no need to previously set a range for $\lambda$)
#         * Cons: Generalize not as good as cross validation; Need strong assumptions on the model; Difficult to mathematically calculate the MAP and the iterative method for computing $\alpha$ and $\beta$ by maximazing the evidence function.
# 
# 
# * **Q: Given these factors, what general conclusions can you make about deciding which model selection method to use?**
# 
#     **A: ** When we want more accurate model and do not concern with run time, we can use cross validation scheme. When we concern with run time, we'd better choose Bayesian model selection scheme.
