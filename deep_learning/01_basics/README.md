# Linear Regression

![alt text](lr.webp "Title")

Linear Regression (LR) is a statistical supervised learning technique to predict the quantitative variable by forming a linear relationship with one or more independent features.

In simple terms, a basic model to solve the problem of Linear Regression (LR) has the following form:

y = β₀ + β₁ * x₁ + ε


Where:
- **y** is the dependent variable (the output or target).
- **x₁** is the independent variable (the input feature).
- **β₀** is the intercept.
- **β₁** is the slope or coefficient.
- **ε** is the error term.

For **multiple linear regression**, the equation looks like this:

y = β₀ + β₁ * x₁ + β₂ * x₂ + ... + βₙ * xₙ + ε


Where:
- **y** is the dependent variable (the output or target).
- **x₁, x₂, ..., xₙ** are the independent variables (the input features).
- **β₀** is the intercept.
- **β₁, β₂, ..., βₙ** are the coefficients for each corresponding feature.
- **ε** is the error term.

## Linear Regression in Deep Learing

However, to simplify everything in the way Deep Learning (DL) works, we call **βᵢ** (with **i** from 1 to **n**) as **wᵢ** (weights), **β₀** as **b** (bias), and we assume **ε** equals 0. Essentially, at this point, we consider the Linear Regression (LR) model as a **Neural Network** with just one layer that contains **wᵢ** and **b** connecting the input and output **y**. In other words, our model now has the form:

y = b + w₁ * x₁ + w₂ * x₂ + ... + wₙ * xₙ 

In this tutorial, to solve a simple problem, our **x** will only have one dimension. Therefore, our model will now have the form:

y = w * x + b

Where:

- **Weights (wᵢ)**: In DL, the coefficients **βᵢ** are often referred to as **weights** because they represent the strength of the connection between the input feature \(x_i\) and the output \(y\).
- **Bias (b)**: **β₀** is called **bias** in DL. The bias helps shift the output up or down, giving the model more flexibility to fit the data.
- **Assuming ε = 0**: In DL, we typically focus on learning the parameters (weights and bias), and the error term **ε** is assumed to be implicitly minimized through optimization techniques like gradient descent.
- **Neural Network Interpretation**: A Linear Regression model can be viewed as a single-layer neural network, where each input **x** is connected to the output **y** via the learned weights **w** and bias **b**.

In summary, when solving a simple LR problem with one input dimension, the equation becomes:

y = w * x + b

This corresponds to a neural network with one input, one output, and no hidden layers, which is effectively a linear mapping from input to output.
