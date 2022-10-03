<<<<<<< HEAD
# Multiple Varialbe Linear Regression(Housing Price Prediction)

from ast import increment_lineno
import copy, math
from re import L
import numpy as np
import matplotlib as plt
plt.style.use('seaborn')
np.set_printoptions(precision=2) # numpy arrays 소수점 제한

# Training Example

X_train = np.array([[2104, 5, 1, 45], [1416,3,2,40], [852,2,1,35]])
Y_train = np.array([460, 232, 178])

# Cost Function 

def compute_cost(X,y,w,b):
    
    m = X_train.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost += (f_wb-y[i])**2
    
    cost = cost/(2*m)
    
    return cost

# Gradient with Multiple Variables

def compute_gradient(X,y,w,b):
    
    m,n = X.shape
    
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db += err
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db  
        
# Gradient Descent With Mulitple Vaiables

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function, alpha, num_iters):
    
    J_history = []
    w = copy.deepcopy(w_in) # w_in의 배열 변경을 피하기 위해 깊은 복사
    b = b_in
    
    for i in range(num_iters):
         
         # 미분값 계산
         dj_dw, dj_db = gradient_function(X,y,w,b)
         
         # Parameters 갱신하기
         w = w - alpha*dj_dw
         b = b - alpha*dj_db
         
         # Parameter 갱신마다 변화된 cost값 저장하기
         if i<100000:
             J_history.append(cost_function(X,y,w,b))
        
         # 100번 단위 반복마다 cost값 인쇄하기
         if i%math.ceil(num_iters / 10) == 0:
             print(f"Iteration:{i:4d}\t Cost:{J_history[-1]:4.2f}")
             
    return w,b,J_history

# Run Gradient Descent 

init_w = np.zeros((4,))
init_b = 0

iterations = 1000
alpha = 5.0e-7

w_f,b_f,J_hist = gradient_descent(X_train,Y_train,init_w,init_b, compute_cost,compute_gradient,alpha,iterations)

print(f"w,b found by gradient descent: {w_f}, {b_f:.2f}")

m = X_train.shape[0]

for i in range(m):
    print(f"Prediction: {np.dot(X_train[i],w_f)+b_f:0.2f}, target: {Y_train[i]})") 

          

            

=======
# Multiple Varialbe Linear Regression(Housing Price Prediction)

from ast import increment_lineno
import copy, math
from re import L
import numpy as np
import matplotlib as plt
plt.style.use('seaborn')
np.set_printoptions(precision=2) # numpy arrays 소수점 제한

# Training Example

X_train = np.array([[2104, 5, 1, 45], [1416,3,2,40], [852,2,1,35]])
Y_train = np.array([460, 232, 178])

# Cost Function 

def compute_cost(X,y,w,b):
    
    m = X_train.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost += (f_wb-y[i])**2
    
    cost = cost/(2*m)
    
    return cost

# Gradient with Multiple Variables

def compute_gradient(X,y,w,b):
    
    m,n = X.shape
    
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db += err
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db  
        
# Gradient Descent With Mulitple Vaiables

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function, alpha, num_iters):
    
    J_history = []
    w = copy.deepcopy(w_in) # w_in의 배열 변경을 피하기 위해 깊은 복사
    b = b_in
    
    for i in range(num_iters):
         
         # 미분값 계산
         dj_dw, dj_db = gradient_function(X,y,w,b)
         
         # Parameters 갱신하기
         w = w - alpha*dj_dw
         b = b - alpha*dj_db
         
         # Parameter 갱신마다 변화된 cost값 저장하기
         if i<100000:
             J_history.append(cost_function(X,y,w,b))
        
         # 100번 단위 반복마다 cost값 인쇄하기
         if i%math.ceil(num_iters / 10) == 0:
             print(f"Iteration:{i:4d}\t Cost:{J_history[-1]:4.2f}")
             
    return w,b,J_history

# Run Gradient Descent 

init_w = np.zeros((4,))
init_b = 0

iterations = 1000
alpha = 5.0e-7

w_f,b_f,J_hist = gradient_descent(X_train,Y_train,init_w,init_b, compute_cost,compute_gradient,alpha,iterations)

print(f"w,b found by gradient descent: {w_f}, {b_f:.2f}")

m = X_train.shape[0]

for i in range(m):
    print(f"Prediction: {np.dot(X_train[i],w_f)+b_f:0.2f}, target: {Y_train[i]})") 

          

            

>>>>>>> 24b7b015b4a80480c949646a1a68724c7c29f93d
