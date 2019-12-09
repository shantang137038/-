import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


data = np.genfromtxt("C:\\Users\\ASUS\\Desktop\\data\\duoyuan\\train.csv", delimiter=',')
x_data = data[1:, 1:-1]
y_data = data[1:, -1]
print(data)
test = np.genfromtxt("C:\\Users\\ASUS\\Desktop\\data\\duoyuan\\test.csv", delimiter=',')
print(test)
x_test = test[1:, 1:-1]
y_test = test[1:, -1]
print(x_test)

lr = 0.0001
theta0 = 0
theta1 = 0
theta2 = 0
epochs = 1000


def computer_error(theta0 , theta1 , theta2 , x_data , y_data):
    total_Error = 0
    for i in range(len(x_data)):
        total_Error += (y_data[i] - theta1 * x_data[i,0] - theta2 * x_data[i,1] - theta0) ** 2
    return total_Error/float(len(x_data))


def gradient_discent_runner(x_data,y_data ,lr, theta0, theta1,theta2 ,epochs):
    m = float(len(x_data))
    #print("111111111111111111",theta0)
    for i in range(0, epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0, len(x_data)):
            theta0_grad += (1/m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) - y_data[j])
            theta1_grad += (1/m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) - y_data[j]) * x_data[j, 0]
            theta2_grad += (1/m) * ((theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1]) - y_data[j]) * x_data[j, 1]
        #print("0000000000000000",theta0_grad)
        theta0 = theta0 - (lr * theta0_grad)
        theta1 = theta1 - (lr * theta1_grad)
        theta2 = theta2 - (lr * theta2_grad)
    #print("theta012,,,====", theta0, theta1, theta2)
    return theta0, theta1, theta2


print("theta0 = {0} , theta1 = {1} , theta2 = {2}, computer_error={3}".format(theta0 , theta1 , theta2, computer_error(theta0 , theta1 , theta2 , x_data , y_data)))
print("Running....")
theta0, theta1, theta2 = gradient_discent_runner(x_data , y_data , lr , theta0 , theta1 , theta2 ,epochs)


print("After Running theta0 = {0} , theta1 = {1} , theta2 = {2}, computer_error={3}".format(theta0 , theta1 , theta2, computer_error(theta0 , theta1 , theta2 , x_data , y_data)))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data,c='r',marker='o',s=100) #点为红色三角形
x0 = x_data[:, 0]
x1 = x_data[:, 1]
x0, x1 = np.meshgrid(x0, x1)
#plt.scatter(x0,x1)
#plt.show()
x_test1 = x_test[:, 0]
x_test2 = x_test[:, 1]
z = theta0 + theta1 * x0 + theta2 * x1
c = theta0 + theta1 * x_test1 + theta2 * x_test2
print(z)
print("predict is ",c)
ax.plot_surface(X=x0,Y=x1,Z=z)
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()

model = linear_model.LinearRegression()
model.fit(x_data,y_data)
x_predict = model.predict(x_test)
print(model.score(x_test,y_test))
print(x_predict)


