from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image = None

    x_values = ""
    y_values = ""

    if request.method == "POST":

        x_values = request.form["x"]
        y_values = request.form["y"]

        try:
            x = np.array(list(map(float, x_values.split(","))))
            y = np.array(list(map(float, y_values.split(","))))

            if len(x) != len(y):
                return render_template(
                    "index.html",
                    error="X and Y must contain the same number of values.",
                    x_values=x_values,
                    y_values=y_values
                )

            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x * x)

            denominator = (n * sum_xx) - (sum_x ** 2)

            if denominator == 0:
                return render_template(
                    "index.html",
                    error="Cannot calculate regression. All X values are identical.",
                    x_values=x_values,
                    y_values=y_values
                )
            m = ((n * sum_xy) - (sum_x * sum_y)) / denominator
            c = (sum_y - (m * sum_x)) / n

            y_pred = m * x + c

            mse = np.mean((y_pred - y) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y))

            if "plot" in request.form:

                if not os.path.exists("static"):
                    os.makedirs("static")

                plt.figure(figsize=(7, 5))

                plt.scatter(
                    x,
                    y,
                    color="red",
                    label="Actual Data"
                )

                plt.plot(
                    x,
                    y_pred,
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
                    "static/regression_plot.png",
                    bbox_inches="tight"
                )

                plt.close()

                image = "regression_plot.png"

            result = {
                "m": round(m, 4),
                "c": round(c, 4),
                "pred": np.round(y_pred, 3),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4)
            }


        except ValueError:

            return render_template(
                "index.html",
                error="Error : Please enter valid numeric values separated by commas.",
                x_values=x_values,
                y_values=y_values
            )


    return render_template(
        "index.html",
        result=result,
        image=image,
        x_values=x_values,
        y_values=y_values
    )



@app.route("/reset")
def reset():

    return render_template(
        "index.html",
        result=None,
        image=None,
        x_values="",
        y_values=""
    )



if __name__ == "__main__":
    app.run(debug=True)