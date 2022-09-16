# Univariate Linear Regression(Housing Price Example)

from re import L
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import copy, math

# Training example

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]) #feature values
y_train = np.array([250, 300, 480, 430, 630, 730]) #targets


# i 번째 훈련 데이터
i=0
x_i = x_train[i] 
y_i = y_train[i]


# 모델 예측값(Prediction) 구하기

def compute_model_output(x,w,b):
    
    m = x.shape[0]
    f_xb = np.zeros(m)
    
    for i in range(m):
        f_xb[i] = w*x[i] + b
        
    return f_xb

# 비용함수

def compute_cost(x,y,w,b):
    
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1/(2*m)) * cost_sum
    
    return total_cost

# Gradient(비용 함수의 한 점에서 편미분값 구하기)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw += (f_wb - y[i])*x[i]
        dj_db += (f_wb - y[i])
    
    return dj_dw, dj_db

# Descent(최적의 parameter 찾기)

def Gradient_Descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    
    w = copy.deepcopy(w_in)
    J_history = []
    P_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            P_history.append([w,b])
            
    return w, b, J_history, P_history
        
# Gradient Desecnt 구동

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, P_hist = Gradient_Descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(w_final, b_final)

#Plotting the date

tmp_f_wb = compute_model_output(x_train,w_final,b_final)
plt.plot(x_train, tmp_f_wb, c='b', label='Prediction')
plt.scatter(x_train, y_train, marker = 'x', c = 'r', label = 'Actual Values')
plt.title("Housing Prices")
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()



