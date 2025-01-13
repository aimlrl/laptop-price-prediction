import numpy as np
from sklearn.metrics import r2_score




def mse(theta0,theta,X_train_transpose,y_train):

    return np.mean((y_train - (theta0 + np.matmul(X_train_transpose,theta)))**2)




def del_mse_by_del_theta(theta0,theta,X_train_transpose,y_train):

    error_transpose = np.transpose((theta0 + np.matmul(X_train_transpose,theta) - y_train))

    del_by_del_theta0 = np.mean(error_transpose)
    del_by_del_theta = (1/y_train.shape[0])*np.transpose(np.matmul(error_transpose,X_train_transpose))

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
    print("Performance on Training Data is {}".format(r2_score(y_true=y_train,y_pred=y_train_pred)))

    return [theta0_final,theta_final]