# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:33:48 2021

@author: vohuynhq
"""
import numpy as np
import PIL
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math

def basic_operations(A,B,C,D,
                  dim_A,dim_B,dim_C,dim_D,
                  shape_A,shape_B,shape_C,shape_D,
                  dtype_A,dtype_B,dtype_C,dtype_D,
                  B_2_2,B_1_all,B_all_3,
                  new_B,
                  B_final_column,
                  min_B, max_B, min_each_row_B, max_each_row_B, min_each_column_B,max_each_column_B,
                  zeros, ones, identity, ver_stacked, hor_stacked,
                  *args):
    
    ##
    # Question 1:
    #
    A_true = np.array([[1, 2, 3], [1, 2, 3]])
    B_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    C_true = np.array([[1.0, 2.0, 3.0]])
    D_true = np.array([[4.0],[5], [6]])
    
    assert A == A_true, "Wrong answer!"
    assert B == B_true, "Wrong answer!"
    assert C == C_true, "Wrong answer!"
    assert D == D_true, "Wrong answer!"
    
    ##
    # Question 2:
    #
    assert dim_A == A_true.ndim, "Wrong answer!"
    assert dim_B == B_true.ndim, "Wrong answer!"
    assert dim_C == C_true.ndim, "Wrong answer!"
    assert dim_D == D_true.ndim, "Wrong answer!"

    ##
    # Question 3:
    #
    assert shape_A == A_true.shape, "Wrong answer!"
    assert shape_B == B_true.shape, "Wrong answer!"
    assert shape_C == C_true.shape, "Wrong answer!"
    assert shape_D == D_true.shape, "Wrong answer!"

    ## 
    # Question 4:
    #
    assert dtype_A == A_true.dtype, "Wrong answer!"
    assert dtype_B == B_true.dtype, "Wrong answer!"
    assert dtype_C == C_true.dtype, "Wrong answer!"
    assert dtype_D == D_true.dtype, "Wrong answer!"


    ##
    # Question 5:
    #
    assert B_2_2 == B_true[1,1], "Wrong answer!"
    assert B_1_all == B_true[0,:], "Wrong answer!"
    assert B_all_3 == B_true[:,2], "Wrong answer!"
        
    
    ##
    # Question 6:
    #
    B_true[1,1] = 1.50
    assert new_B == B_true, "Wrong answer!"
    
    ##
    # Questiom 7:
    #
    assert B_final_column == B_true[:, -1], "Wrong answer!"
    
    
    ##
    # Question 8:
    #
    assert min_B == np.min(B), "Wrong answer!"
    assert max_B == np.min(B), "Wrong answer!"
    assert min_each_row_B ==  np.min(B,axis=1), "Wrong answer!"
    assert max_each_row_B == np.max(B,axis =1), "Wrong answer!"
    assert min_each_column_B == np.min(B,axis=0), "Wrong answer!"
    assert max_each_column_B == np.max(B,axis=0), "Wrong answer!"

    ## 
    # Question 9:
    #
    
    A_zeros = np.zeros(shape=(10, 10))
    A_ones = np.ones(shape=(10, 10))
    A_identity = np.identity(n=10)
    vertically_stack = np.vstack([A_zeros, A_ones, A_identity])
    horizontally_stack = np.hstack([A_zeros, A_ones, A_identity])
    
    assert zeros == A_zeros, "Wrong answer!"
    assert ones == A_ones, "Wrong answer!"    
    assert identity == A_identity, "Wrong answer!"
    assert ver_stacked == vertically_stack, "Wrong answer!"
    assert hor_stacked == horizontally_stack, "Wrong answer!"
    
    return None

def mathematical_operations(A_plus_2, B_minus_5, A_over_2, C_over_2, D_times_2, D_squared,
                            sin_A, cos_A, tan_A, ln_A, log_10_A,
                            A_dot_B, C_dot_D, A_dot_C,
                            det_B, eig_vals_B, eig_vecs_B, inv_B,
                            *args):
                            
    A_true = np.array([[1, 2, 3], [1, 2, 3]])
    B_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    C_true = np.array([[1.0, 2.0, 3.0]])
    D_true = np.array([[4.0],[5], [6]])
    
    ##
    # Question 1:
    #
    assert A_plus_2 == (A_true + 2), "Wrong answer!" 
    assert B_minus_5 == (B_true - 5) , "Wrong answer!"
    assert A_over_2 == (A_true / 2), "Wrong answer!"
    assert C_over_2 == (C_true / 2), "Wrong answer!"
    assert D_times_2 == (D_true * 2), "Wrong answer!"
    assert D_squared == (np.square(D_true)), "Wrong answer!"
    assert sin_A == (np.sin(A_true)), "Wrong answer!"
    assert cos_A == (np.cos(A_true)), "Wrong answer!"
    assert tan_A == (np.tan(A_true)), "Wrong answer!"
    assert ln_A == (np.log(A_true)), "Wrong answer!"
    assert log_10_A == (np.log10(A_true)), "Wrong answer!"
    
    ##
    # Question 2:
    #
    assert A_dot_B == (np.dot(A_true,B_true)), "Wrong answer!"
    assert C_dot_D == (np.dot(C_true,B_true)), "Wrong answer!"
    
    ##
    # Question 3:
    #
    assert det_B == (np.linalg.det(B_true)), "Wrong answer!"
    assert eig_vals_B == (np.linalg.eig(B_true)[0]), "Wrong answer!"
    assert eig_vecs_B == (np.linalg.eig(B_true)[1]), "Wrong answer!"
    assert inv_B == (np.linalg.inv(B_true)), "Wrong answer!"
    
    return None

def system_of_equations(x_1,x_2,*args):
    
    ##
    # Question 1:
    #
    A = np.array([[3, -2], [2, -2]])
    B = np.array([[7], [2]])
    x_1_true = np.linalg.solve(A, B)
    assert x_1 == x_1_true, "Wrong answer!"
    
    ##
    # Question 2:
    #
    A = np.array([[3, -2, 1], [1, 1, 1], [3, -2, -1]])
    B = np.array([[7], [2], [3]])
    x_2_true = np.linalg.solve(A, B)
    assert x_2 == x_2_true, "Wrong answer!"
    
    return None
