import numpy as np
import matplotlib.pyplot as plt

choice = int(input("Enter 1 for Ridge Regression or 2 for Lasso Regression: "))

if choice == 1:

    def predict(x, w, b):
        return w * x + b

    def calc_error(y, y_pred):
        return y_pred - y

    def ridge_cost(x, y, w, b, lamda):
        n = len(x)
        y_pred = predict(x, w, b)
        error = calc_error(y, y_pred)
        cost = (1 / (2 * n)) * sum(error ** 2)
        penalty = (lamda / 2) * (w ** 2)
        return cost + penalty

    def calc_ridge_gradient(x, y, w, b, lamda):
        prediction = predict(x, w, b)
        error = calc_error(y, prediction)
        n = len(x)
        dw = (1 / n) * sum(x * error) + lamda * w
        db = (1 / n) * sum(error)
        return dw, db

    def update_ridge_parameter(w, b, dw, db, alpha):
        w = w - alpha * dw
        b = b - alpha * db
        return w, b

    x = np.array([1, 2, 3, 4])
    y = np.array([2, 2, 4, 5])

    w = 0
    b = 0
    alpha = 0.05
    lamda = 0.2
    iteration = 4
    cost_history = []

    for i in range(iteration):
        predictions = predict(x, w, b)
        dw, db = calc_ridge_gradient(x, y, w, b, lamda)
        w, b = update_ridge_parameter(w, b, dw, db, alpha)
        cost = ridge_cost(x, y, w, b, lamda)
        cost_history.append(cost)

        print("Iteration:", i + 1)
        print("Predictions:", predictions)
        print("Gradient w.r.t W:", dw)
        print("Gradient w.r.t b:", db)
        print("Updated W:", w)
        print("Updated b:", b)
        print("Cost:", cost)
        print("------------------------------")

    iterations = np.arange(1, len(cost_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.scatter(iterations, cost_history)
    plt.plot(iterations, cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Ridge Regression: Cost vs Iteration")
    plt.show()


elif choice == 2:

    def predict(x1, x2, w1, w2, b):
        return w1 * x1 + w2 * x2 + b

    def calc_error(y, y_pred):
        return y_pred - y

    def lasso_cost(x1, x2, y, w1, w2, b, lamda):
        n = len(x1)
        y_pred = predict(x1, x2, w1, w2, b)
        error = calc_error(y, y_pred)
        cost = (1 / (2 * n)) * sum(error ** 2)
        penalty = lamda * (abs(w1) + abs(w2))
        return cost + penalty

    def calc_lasso_gradient(x1, x2, y, w1, w2, b, lamda):
        n = len(x1)
        prediction = predict(x1, x2, w1, w2, b)
        error = calc_error(y, prediction)
        dw1 = (1 / n) * sum(x1 * error) + lamda * np.sign(w1)
        dw2 = (1 / n) * sum(x2 * error) + lamda * np.sign(w2)
        db = (1 / n) * sum(error)
        return dw1, dw2, db

    def update_lasso_parameter(w1, w2, b, dw1, dw2, db, alpha):
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        b = b - alpha * db
        return w1, w2, b

    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([2, 4, 6, 8])
    y = np.array([2, 4, 6, 8])

    w1 = 0
    w2 = 0
    b = 0
    alpha = 0.05
    lamda = 0.2
    iteration = 4
    cost_history = []

    for i in range(iteration):
        predictions = predict(x1, x2, w1, w2, b)
        dw1, dw2, db = calc_lasso_gradient(
            x1, x2, y, w1, w2, b, lamda
        )
        w1, w2, b = update_lasso_parameter(
            w1, w2, b, dw1, dw2, db, alpha
        )
        cost = lasso_cost(
            x1, x2, y, w1, w2, b, lamda
        )
        cost_history.append(cost)

        print("Iteration:", i + 1)
        print("Predictions:", predictions)
        print("Gradient w.r.t W1:", dw1)
        print("Gradient w.r.t W2:", dw2)
        print("Gradient w.r.t b:", db)
        print("Updated W1:", w1)
        print("Updated W2:", w2)
        print("Updated b:", b)
        print("Cost:", cost)
        print("------------------------------")

    iterations = np.arange(1, len(cost_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.scatter(iterations, cost_history)
    plt.plot(iterations, cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Lasso Regression: Cost vs Iteration")
    plt.show()

else:
    print("Invalid choice! Please enter 1 or 2.")