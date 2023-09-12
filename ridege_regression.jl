using CSV
using LinearAlgebra
using DataFrames
using Plots

# Load data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataframe = CSV.File(download(url); delim=";") |> DataFrame
data = Matrix(dataframe)

# Split data into train and test
train_data = data[1:1400, :]
test_data = data[1401:end, :]

# Separate features and labels
X_train = [ones(size(train_data, 1)) train_data[:, 1:11]] 
y_train = train_data[:, 12]
X_test = [ones(size(test_data, 1)) test_data[:, 1:11]]
y_test = test_data[:, 12]

# Define ridge regression function
function ridge_regression(Phi, y, lambda)
    return [Phi ; sqrt(lambda)*Matrix{Float64}(I, size(Phi,2), size(Phi,2))] \ ([y ; zeros(size(Phi,2))])
end

# Define MSE function
function compute_mse(y_pred, y_true)
    return norm(y_pred - y_true)^2/length(y_true)
end


lambdas = [0, 1e-3, 1e-2, 1e-1]
errors = []

for lambda in lambdas
    theta = ridge_regression(X_train, y_train, lambda)
    y_pred = X_test * theta
    mse = compute_mse(y_pred, y_test)
    push!(errors, mse)
    println("MSE for lambda = $lambda: $mse")
end

plot(lambdas, errors, label="MSE", xlabel="lambda", ylabel="MSE", title="MSE vs lambda", marker=:circle, lw=2)
savefig("error_vs_lambda.png")