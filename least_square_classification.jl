# regularized least square classification for https://archive.ics.uci.edu/dataset/52/ionosphere

using CSV
using LinearAlgebra
using DataFrames
using Plots

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
dataframe = CSV.File(download(url); delim=",", header=false) |> DataFrame
data = Matrix(dataframe)

# convert labels to 1 and -1 from g and b
data[:, 35] = [entry == "g" ? 1.0 : -1.0 for entry in data[:, 35]]

# make sure all values are float
data = map(Float64, data)

# Split data into train and test
train_data = data[1:300, :]
test_data = data[301:end, :]

# Separate features and labels
X_train = [ones(size(train_data, 1)) train_data[:, 1:34]] 
y_train = train_data[:, 35]
X_test = [ones(size(test_data, 1)) test_data[:, 1:34]]
y_test = test_data[:, 35]

# Define ridge regression function
function ridge_regression(Phi, y, lambda)
    return [Phi ; sqrt(lambda)*Matrix{Float64}(I, size(Phi,2), size(Phi,2))] \ ([y ; zeros(size(Phi,2))])
end

# Define calculate accuracy 
function compute_accuracy(y_pred, y_true)
    correct_predictions = sum(y_pred .== y_true)
    return correct_predictions / length(y_true)
end


lambdas = [0, 1e-3, 1e-2, 1e-1]
accuracy_list = []

for lambda in lambdas
    theta = ridge_regression(X_train, y_train, lambda)
    y_pred = sign.(X_test * theta)
    accuracy = compute_accuracy(y_pred, y_test)
    push!(accuracy_list, accuracy)
    println("Prediction accuracy for lambda = $lambda: $accuracy")
end

plot(lambdas, accuracy_list, label="accuracy", xlabel="lambda", ylabel="accuracy", title="accuracy vs lambda", marker=:circle, lw=2)
savefig("accuracy_vs_lambda.png")