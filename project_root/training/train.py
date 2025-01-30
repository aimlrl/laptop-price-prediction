import numpy as np
from sklearn.metrics import r2_score
from config import config




def mse(theta0,theta,X_train_transpose,y_train):

    if config.PENALTY == "l1":
        return np.mean((y_train - (theta0 + np.matmul(X_train_transpose,theta)))**2) +\
              ((config.LAMBDA*np.sum(np.abs(theta)))/y_train.shape[0])
    
    elif config.PENALTY == "l2":
        return np.mean((y_train - (theta0 + np.matmul(X_train_transpose,theta)))**2) +\
              ((config.LAMBDA*np.sum(theta**2))/(2 * y_train.shape[0]))
    
    else:
        return np.mean((y_train - (theta0 + np.matmul(X_train_transpose,theta)))**2)

    



def del_mse_by_del_theta(theta0,theta,X_train_transpose,y_train):

    error_transpose = np.transpose((theta0 + np.matmul(X_train_transpose,theta) - y_train))

    del_by_del_theta0 = np.mean(error_transpose)

    if config.PENALTY == "l1":
        del_by_del_theta = (1/y_train.shape[0])*(np.transpose(np.matmul(error_transpose,X_train_transpose)) +\
                                                 config.LAMBDA)
        
    elif config.PENALTY == "l2":
        del_by_del_theta = (1/y_train.shape[0])*(np.transpose(np.matmul(error_transpose,X_train_transpose)) +\
                                                 (config.LAMBDA*theta))
        
    else:
        del_by_del_theta = (1/y_train.shape[0])*(np.transpose(np.matmul(error_transpose,X_train_transpose)))

    return [del_by_del_theta0,del_by_del_theta]




def training(epsilon,X_train_transpose,y_train,tol):

    epoch_counter = 0
    theta0_initial = 0
    theta_initial = np.zeros((X_train_transpose.shape[1],1))

    while True:

        initial_gradients = del_mse_by_del_theta(theta0_initial,theta_initial,X_train_transpose,y_train)

        theta0_final = theta0_initial - (epsilon * initial_gradients[0])
        theta_final = theta_initial - (epsilon * initial_gradients[1])

        mse_initial_value = mse(theta0_initial,theta_initial,X_train_transpose,y_train)
        mse_final_value = mse(theta0_final,theta_final,X_train_transpose,y_train)

        if abs(mse_initial_value - mse_final_value) < tol:
            break

        epoch_counter += 1

        theta0_initial = theta0_final
        theta_initial = theta_final

        print("Epoch # {}, MSE Value = {}".format(epoch_counter,mse_initial_value))

    y_train_pred = theta0_final + np.matmul(X_train_transpose,theta_final)
    print("\nPerformance on Training Data is {}".format(r2_score(y_true=y_train,y_pred=y_train_pred)))

    return [theta0_final,theta_final]
