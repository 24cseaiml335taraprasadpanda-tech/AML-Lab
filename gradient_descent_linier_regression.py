import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
x=np.array([1,2,3])
y=np.array([2,3,4])
m=len(x)
n=0.05
b=0
w=0
iteration=2
print("initial Weight: ",w)
print("initial bias: ",b)
ypredictions=[]
for i in range(m):
  ypredictions.append(w*x[i]+b)
loss=mean_squared_error(y,ypredictions)
print("initial cost: ",loss)
cost1=[]
cost1.append(loss)
for i in range(iteration):
    print("Iteration: ",i+1)
    y_pred=w*x+b
    dw=-(2/m)*sum(x*(y-y_pred))
    db=-(2/m)*sum(y-y_pred)
    print("Gradient w.r.t weight: ",dw)
    print("Gradient w.r.t bias: ",db)
    w=w-n*dw
    b=b-n*db
    print("Updated weight: ",w)
    print("Updated bias: ",b)
    print("New regression equation: ")
    print("y = ",w,"x + ",b)
    ypred=[]
    for i in range(m):
      ypred.append(w*x[i]+b)
    print("the updated prediction values: ",ypred)
    cost=[]
    for i in range(m):
      cost.append((y[i]-ypred[i])**2)
    total_cost=np.average(cost)
    cost1.append(total_cost)
    cost.clear()
    print("the total cost: ",total_cost)
    print(" value of the gradients at itteration : ")
    print("w = ",w,"b = ",b,"cost = ",total_cost)
cost1=np.array(cost1)
plt.figure(figsize=(10, 5))
plt.scatter(x, y,color="blue",label="Training Data")
plt.plot(x, ypred,color="red",label="Regration Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training Data and Regression Line")
plt.legend()
plt.show()
print(cost1)
plt.figure(figsize=(10, 5))
iterations = np.arange(len(cost1))
plt.scatter(iterations, cost1)
plt.plot(iterations, cost1)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iteration")
plt.show()