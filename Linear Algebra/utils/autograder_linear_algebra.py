# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:33:48 2021

@author: vohuynhq
"""

import numpy as np

def example(A):
    A_true = np.array([[3, 4, 5], [3, 4, 5]])
    np.testing.assert_equal(A, A_true)
    return None

def basic_operations(A,B,C,D,
                     dim_A,dim_B,dim_C,dim_D,
                     shape_A,shape_B,shape_C,shape_D,
                     dtype_A,dtype_B,dtype_C,dtype_D,
                     B_2_2,B_1_all,B_all_3,
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
    
    np.testing.assert_equal(A, A_true)
    np.testing.assert_equal(B, B_true)
    np.testing.assert_equal(C, C_true)
    np.testing.assert_equal(D, D_true)
    
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
    np.testing.assert_equal(B_1_all,B_true[0,:])
    np.testing.assert_equal(B_all_3,B_true[:,2])
    
        
    ##
    # Question 6:
    #
    assert min_B == np.min(B), "Wrong answer!"
    assert max_B == np.max(B), "Wrong answer!"
    np.testing.assert_equal(min_each_row_B,np.min(B,axis=1))
    np.testing.assert_equal(max_each_row_B,np.max(B,axis =1))
    np.testing.assert_equal(min_each_column_B,np.min(B,axis=0))
    np.testing.assert_equal(max_each_column_B,np.max(B,axis=0))
    
    
    ## 
    # Question 7:
    #
    
    A_zeros = np.zeros(shape=(10, 10))
    A_ones = np.ones(shape=(10, 10))
    A_identity = np.identity(n=10)
    vertically_stack = np.vstack([A_zeros, A_ones, A_identity])
    horizontally_stack = np.hstack([A_zeros, A_ones, A_identity])
    
    np.testing.assert_equal(zeros, A_zeros)
    np.testing.assert_equal(ones, A_ones)
    np.testing.assert_equal(identity, A_identity)
    np.testing.assert_equal(ver_stacked,vertically_stack)
    np.testing.assert_equal(hor_stacked,horizontally_stack)
    
    return None

def mathematical_operations(A_plus_2, B_minus_5, A_over_2, C_over_2, D_times_2, D_squared,
                            sin_A, cos_A, tan_A, ln_A, log_10_A,
                            A_dot_B, C_dot_D,
                            det_B, eig_vals_B, eig_vecs_B, inv_B,
                            *args):
                            
    A_true = np.array([[1, 2, 3], [1, 2, 3]])
    B_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    C_true = np.array([[1.0, 2.0, 3.0]])
    D_true = np.array([[4.0],[5], [6]])
    
    ##
    # Question 1:
    #
    np.testing.assert_equal(A_plus_2,(A_true + 2))
    np.testing.assert_equal(B_minus_5,(B_true - 5))
    np.testing.assert_equal(A_over_2,(A_true / 2))
    np.testing.assert_equal(C_over_2,(C_true / 2))
    np.testing.assert_equal(D_times_2,(D_true * 2))
    np.testing.assert_equal(D_squared,(np.square(D_true)))
    np.testing.assert_equal(sin_A,(np.sin(A_true)))
    np.testing.assert_equal(cos_A,(np.cos(A_true)))
    np.testing.assert_equal(tan_A,(np.tan(A_true)))
    np.testing.assert_equal(ln_A,(np.log(A_true)))
    np.testing.assert_equal(log_10_A,(np.log10(A_true))),
    
    ##
    # Question 2:
    #
    np.testing.assert_equal(A_dot_B,(np.dot(A_true,B_true)))
    np.testing.assert_equal(C_dot_D,(np.dot(C_true,D_true)))
    
    ##
    # Question 3:
    #
    np.testing.assert_equal(det_B,(np.linalg.det(B_true)))
    np.testing.assert_equal(eig_vals_B,(np.linalg.eig(B_true)[0]))
    np.testing.assert_equal(eig_vecs_B,(np.linalg.eig(B_true)[1]))
    np.testing.assert_equal(inv_B,(np.linalg.inv(B_true)))
    
    return None

def system_of_equations(x_1,x_2,*args):
    
    ##
    # Question 1:
    #
    A = np.array([[3, -2], [2, -2]])
    B = np.array([[7], [2]])
    x_1_true = np.linalg.solve(A, B)
    np.testing.assert_equal(x_1,x_1_true)
    
    ##
    # Question 2:
    #
    A = np.array([[3, -2, 1], [1, 1, 1], [3, -2, -1]])
    B = np.array([[7], [2], [3]])
    x_2_true = np.linalg.solve(A, B)
    np.testing.assert_equal(x_2,x_2_true)
    
    return None

def scalar_vector_projections(scal_proj_v_r,vec_proj_v_r,
                              *args):
    
    r = np.array([[1, 0, 3]])
    s = np.array([[-1], [4], [2]])
    scalar_projection = r.dot(s)/np.linalg.norm(r)
    vector_projection = (r.dot(s)/np.square(np.linalg.norm(r)))*r
    
    np.testing.assert_equal(scal_proj_v_r,scalar_projection)
    np.testing.assert_equal(vec_proj_v_r,vector_projection)
    
    return None

def changing_basis(v_1,v_2, *args):
    
    A = np.array([[1, 1], [1, -1]])
    B = np.array([[5], [-1]])
    v_1_true = np.linalg.solve(A, B)
    np.testing.assert_equal(v_1,v_1_true)
    
    A = np.array([[2, 1, -1], [1, -2, 2], [0, -1, -5]])
    B = np.array([[1], [1], [1]])
    v_2_true = np.linalg.solve(A, B)
    np.testing.assert_equal(v_2,v_2_true)
    
    return None

def transformation_matrix(p1x,p2x,p3x,p4x,
                          p1y,p2y,p3y,p4y,
                          p1xy,p2xy,p3xy,p4xy,
                          p1hx,p2hx,p3hx,p4hx,
                          p1hy,p2hy,p3hy,p4hy,
                          p1hxy,p2hxy,p3hxy,p4hxy,
                          p1c,p2c,p3c,p4c,
                          p1cc,p2cc,p3cc,p4cc,
                          p1tx,p2tx,p3tx,p4tx,
                          p1ty,p2ty,p3ty,p4ty,
                          p1txy,p2txy,p3txy,p4txy,                          
                          *args):
    
    p1 = np.array([[0], [0]])
    p2 = np.array([[0], [4]])
    p3 = np.array([[4], [4]])
    p4 = np.array([[4], [0]])
    
    ##
    # Question 1: 
    #
    kx = 2
    Sx = np.array([[kx, 0], [0, 1]]) 
    p1x_true = Sx.dot(p1)
    p2x_true = Sx.dot(p2)
    p3x_true = Sx.dot(p3)
    p4x_true = Sx.dot(p4)
    
    np.testing.assert_equal(p1x,p1x_true)
    np.testing.assert_equal(p2x,p2x_true)
    np.testing.assert_equal(p3x,p3x_true)
    np.testing.assert_equal(p4x,p4x_true)
    ###
    ky = 2
    Sy = np.array([[1, 0], [0, ky]]) 
    p1y_true = Sy.dot(p1)
    p2y_true = Sy.dot(p2)
    p3y_true = Sy.dot(p3)
    p4y_true = Sy.dot(p4)
    
    np.testing.assert_equal(p1y,p1y_true)
    np.testing.assert_equal(p2y,p2y_true)
    np.testing.assert_equal(p3y,p3y_true)
    np.testing.assert_equal(p4y,p4y_true)
    ###
    kx = 2
    ky = 2
    Sxy = np.array([[kx, 0], [0, ky]]) 
    p1xy_true = Sxy.dot(p1)
    p2xy_true = Sxy.dot(p2)
    p3xy_true = Sxy.dot(p3)
    p4xy_true = Sxy.dot(p4)

    np.testing.assert_equal(p1xy,p1xy_true)
    np.testing.assert_equal(p2xy,p2xy_true)
    np.testing.assert_equal(p3xy,p3xy_true)
    np.testing.assert_equal(p4xy,p4xy_true)
    
    ##
    # Question 2:
    #
    kx = 2
    Shx = np.array([[1, kx], [0, 1]]) 
    p1hx_true = Shx.dot(p1)
    p2hx_true = Shx.dot(p2)
    p3hx_true = Shx.dot(p3)
    p4hx_true = Shx.dot(p4)
    
    np.testing.assert_equal(p1hx,p1hx_true)
    np.testing.assert_equal(p2hx,p2hx_true)
    np.testing.assert_equal(p3hx,p3hx_true)
    np.testing.assert_equal(p4hx,p4hx_true)
    ###
    ky = 2
    Shy = np.array([[1, 0], [ky, 1]]) 
    p1hy_true = Shy.dot(p1)
    p2hy_true = Shy.dot(p2)
    p3hy_true = Shy.dot(p3)
    p4hy_true = Shy.dot(p4)
    
    np.testing.assert_equal(p1hy,p1hy_true)
    np.testing.assert_equal(p2hy,p2hy_true)
    np.testing.assert_equal(p3hy,p3hy_true)
    np.testing.assert_equal(p4hy,p4hy_true)
    ###
    kx = 2
    ky = 2
    Shxy = np.array([[1, kx], [ky, 1]]) 
    p1hxy_true = Shxy.dot(p1)
    p2hxy_true = Shxy.dot(p2)
    p3hxy_true = Shxy.dot(p3)
    p4hxy_true = Shxy.dot(p4)
    
    np.testing.assert_equal(p1hxy,p1hxy_true)
    np.testing.assert_equal(p2hxy,p2hxy_true)
    np.testing.assert_equal(p3hxy,p3hxy_true)
    np.testing.assert_equal(p4hxy,p4hxy_true)

    ##
    # Question 3:
    #

    theta_degree = 90
    theta_rad = theta_degree * np.pi / 180.0
    Rc = np.array([[np.cos(theta_rad), np.sin(theta_rad)], [-np.sin(theta_rad), np.cos(theta_rad)]]) 
    p1c_true = Rc.dot(p1)
    p2c_true = Rc.dot(p2)
    p3c_true = Rc.dot(p3)
    p4c_true = Rc.dot(p4)
    
    np.testing.assert_equal(p1c,p1c_true)
    np.testing.assert_equal(p2c,p2c_true)
    np.testing.assert_equal(p3c,p3c_true)
    np.testing.assert_equal(p4c,p4c_true)

    ###
    theta_degree = 90
    theta_rad = theta_degree * np.pi / 180.0
    Rcc = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]) 
    p1cc_true = Rcc.dot(p1)
    p2cc_true = Rcc.dot(p2)
    p3cc_true = Rcc.dot(p3)
    p4cc_true = Rcc.dot(p4)
    
    np.testing.assert_equal(p1cc,p1cc_true)
    np.testing.assert_equal(p2cc,p2cc_true)
    np.testing.assert_equal(p3cc,p3cc_true)
    np.testing.assert_equal(p4cc,p4cc_true)

    ##
    # Question 4:
    #  
    p1 = np.array([[0], [0], [1]])
    p2 = np.array([[0], [4], [1]])
    p3 = np.array([[4], [4], [1]])
    p4 = np.array([[4], [0], [1]])

    kx = 2
    Tx = np.array([[1, 0, kx], [0, 1, 0]]) 
    p1tx_true = Tx.dot(p1)
    p2tx_true = Tx.dot(p2)
    p3tx_true = Tx.dot(p3)
    p4tx_true = Tx.dot(p4)

    np.testing.assert_equal(p1tx,p1tx_true)
    np.testing.assert_equal(p2tx,p2tx_true)
    np.testing.assert_equal(p3tx,p3tx_true)
    np.testing.assert_equal(p4tx,p4tx_true)
    ###
    ky = 2
    Ty = np.array([[1, 0, 0], [0, 1, ky]]) 
    p1ty_true = Ty.dot(p1)
    p2ty_true = Ty.dot(p2)
    p3ty_true = Ty.dot(p3)
    p4ty_true = Ty.dot(p4)

    np.testing.assert_equal(p1ty,p1ty_true)
    np.testing.assert_equal(p2ty,p2ty_true)
    np.testing.assert_equal(p3ty,p3ty_true)
    np.testing.assert_equal(p4ty,p4ty_true)
    ###
    kx = 2
    ky = 2
    Txy = np.array([[1, 0, kx], [0, 1, ky]]) 
    p1txy_true = Txy.dot(p1)
    p2txy_true = Txy.dot(p2)
    p3txy_true = Txy.dot(p3)
    p4txy_true = Txy.dot(p4) 
    
    np.testing.assert_equal(p1txy,p1txy_true)
    np.testing.assert_equal(p2txy,p2txy_true)
    np.testing.assert_equal(p3txy,p3txy_true)
    np.testing.assert_equal(p4txy,p4txy_true)
    return None