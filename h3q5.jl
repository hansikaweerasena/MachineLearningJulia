# regularized least square classification for https://archive.ics.uci.edu/dataset/52/ionosphere

using CSV
using LinearAlgebra
using DataFrames
using Plots
using Convex, SCS

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
X_train = [train_data[:, 1:34] ones(size(train_data, 1))] 
y_train = train_data[:, 35]
X_test = [test_data[:, 1:34] ones(size(test_data, 1))]
y_test = test_data[:, 35]


# Define calculate accuracy 
function compute_accuracy(y_pred, y_true)
    correct_predictions = sum(y_pred .== y_true)
    return correct_predictions / length(y_true)
end

# Logistic loss function
function custom_logistic_loss(t, y_train)
    return log(1 + exp(-t * y_train))
end

# Hinge loss function
function custom_hinge_loss(t, y_train)
    return max(0, 1 - t * y_train)
end


println("Calculating parameters using least square loss function")
theta = (Diagonal(y_train)*X_train)\ones(size(X_train, 1))
y_pred = sign.(X_test * theta)
accuracy = compute_accuracy(y_pred, y_test)
println("Prediction accuracy when using least square loss function: $accuracy")


# Calculating parameters using hinge loss function
println("Calculating parameters using logistic loss function")
theta_logistic = Variable(size(X_train, 2))
t = X_train * theta_logistic
loss = sum(custom_logistic_loss(t[i], y_train[i]) for i in 1:size(X_train, 1))
problem = minimize(loss)
solve!(problem, SCS.Optimizer, silent_solver = true)
accuracy = compute_accuracy(y_pred, y_test)
println("Prediction accuracy when using logistic loss function: $accuracy")


# Calculating parameters using hinge loss function
println("Calculating parameters using hinge loss function")
theta_hinge = Variable(size(X_train, 2))
t = X_train * theta_hinge
loss = sum(custom_hinge_loss(t[i], y_train[i]) for i in 1:size(X_train, 1))
problem = minimize(loss)
solve!(problem, SCS.Optimizer, silent_solver = true)
accuracy = compute_accuracy(y_pred, y_test)
println("Prediction accuracy when using hinge loss function: $accuracy")
