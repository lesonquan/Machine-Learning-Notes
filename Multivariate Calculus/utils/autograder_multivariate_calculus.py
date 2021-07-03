# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:09:14 2021

@author: vohuynhq
"""

import numpy as np
import pandas as pd
from sympy import symbols

def example(A):
    A_true = np.array([[3, 4, 5], [3, 4, 5]])
    np.testing.assert_equal(A, A_true)
    return None

def derivative():
    x = symbols('x')
    
    return None

data = pd.read_csv("Xy_dataset.csv")
### 
X = np.array(data['X'])
y = np.array(data['y'])
###
X = X.reshape(X.shape[0],1)
y = y.reshape(y.shape[0],1)

def question1(m,b,X,y,*args):
    
    ##
    # Step 1: Initialize m and b:
    #
    m_true = 0
    b_true = 0
    
    for i in range(50):   
    
        ##
        # Step 2: Find y_pred = mx + b:
        #
        y_pred = m*X + b
        
        ##
        # Step 3: Update m and b using the Gradient Descent algorithm:
        #
        dm = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)
        m_true = m_true - 0.1*dm
        b_true = b_true - 0.1*db
       
    np.testing.assert_equal(m,m_true)
    np.testing.assert_equal(b,b_true)
    return None

def question2(m,b,X,y,*args):
    
    ##
    # Step 1: Initialize m and b:
    #
    m_true = 0
    b_true = 0
    
    for i in range(50):   
    
        ##
        # Step 2: Find y_pred = mx + b:
        #
        y_pred = m*X + b
        
        ##
        # Step 3: Update m and b using the Gradient Descent algorithm:
        #
        dm = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)
        m_true = m_true - 0.1*dm
        b_true = b_true - 0.1*db
       
    np.testing.assert_equal(m,m_true)
    np.testing.assert_equal(b,b_true)
    return None

def question3(m,b,X,y,*args):
        
    ##
    # Step 1: Initialize m and b:
    #
    m_true = 0
    b_true = 0
    
    for i in range(50):   
        ##
        # Step 2: Find y_pred = mx + b:
        #
        y_pred = m*X + b
        
        ##
        # Step 3: Update m and b using the Gradient Descent algorithm:
        #
        dm = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)
        m_true = m_true - 0.2*dm
        b_true = b_true - 0.2*db
       
    np.testing.assert_equal(m,m_true)
    np.testing.assert_equal(b,b_true)

    return None

def question4(m,b,X,y,*args):
        
    ##
    # Step 1: Initialize m and b:
    #
    m_true = 0
    b_true = 0
    
    for i in range(100):   
        ##
        # Step 2: Find y_pred = mx + b:
        #
        y_pred = m*X + b
        
        ##
        # Step 3: Update m and b using the Gradient Descent algorithm:
        #
        dm = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)
        m_true = m_true - 0.1*dm
        b_true = b_true - 0.1*db
       
    np.testing.assert_equal(m,m_true)
    np.testing.assert_equal(b,b_true)
    
    return None