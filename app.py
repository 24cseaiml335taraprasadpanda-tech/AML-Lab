from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image = None
    error = None

    x_values = ""
    y_values = ""

    if request.method == "POST":

        action = request.form.get("action")

        x_values = request.form.get("x")
        y_values = request.form.get("y")

        try:

            x = np.array(list(map(float, x_values.split(","))))
            y = np.array(list(map(float, y_values.split(","))))

            if len(x) != len(y):
                error = "X and Y must contain the same number of values."

            elif len(x) < 2:
                error = "Please enter at least two data points."

            else:

                n = len(x)

                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_xx = np.sum(x * x)

                denominator = (n * sum_xx) - (sum_x ** 2)

                if np.isclose(denominator, 0):
                    error = "Cannot calculate regression. All X values are identical."

                else:

                    m = ((n * sum_xy) - (sum_x * sum_y)) / denominator
                    c = (sum_y - (m * sum_x)) / n

                    y_pred = m * x + c

                    mse = np.mean((y - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y - y_pred))

                    result = {
                        "m": round(m, 4),
                        "c": round(c, 4),
                        "pred": np.round(y_pred, 3).tolist(),
                        "mse": round(mse, 4),
                        "rmse": round(rmse, 4),
                        "mae": round(mae, 4)
                    }

                    if action == "visualize graph":

                        if not os.path.exists("static"):
                            os.makedirs("static")

                        image_path = os.path.join(
                            "static",
                            "regression_plot.png"
                        )

                        plt.figure(figsize=(7, 5))

                        plt.scatter(
                            x,
                            y,
                            color="red",
                            label="Actual Data"
                        )

                        idx = np.argsort(x)

                        plt.plot(
                            x[idx],
                            y_pred[idx],
                            color="blue",
                            linewidth=2,
                            label="Regression Line"
                        )

                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.title("Linear Regression")
                        plt.grid(True)
                        plt.legend()

                        plt.savefig(
                            image_path,
                            bbox_inches="tight"
                        )

                        plt.close()

                        image = "regression_plot.png"

        except ValueError:
            error = "Please enter valid numeric values separated by commas."

    return render_template(
        "index.html",
        result=result,
        image=image,
        error=error,
        x_values=x_values,
        y_values=y_values
    )


@app.route("/reset")
def reset():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)