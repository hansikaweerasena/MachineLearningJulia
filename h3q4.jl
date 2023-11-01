using CSV
using LinearAlgebra
using DataFrames
using Plots
using Convex, SCS

# Load data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataframe = CSV.File(download(url); delim=";") |> DataFrame
data = Matrix(dataframe)

# Split data into train and test
train_data = data[1:1400, :]
test_data = data[1401:end, :]

# Separate features and labels (for X_train and X_test the last column is added as a column of ones to represent the bias term (beta))
X_train = [train_data[:, 1:11] ones(size(train_data, 1))] 
y_train = train_data[:, 12]
X_test = [test_data[:, 1:11] ones(size(test_data, 1))]
y_test = test_data[:, 12]


# Define MAE function
function compute_mae(y_pred, y_true)
    return sum(abs.(y_pred - y_true)) / length(y_true)
end

# Huber loss function
function custom_huber_loss(t, threshold=0.25)
    return max(square(t)*(t + 1/2)*(1/2 -t), (abs(t) - threshold)*(-(1/2 + t))*(1/2 - t))
end

# Hinge loss function
function custom_hinge_loss(residual, threshold=0.5)
    return max(0, abs(residual) - threshold)
end

#Calculating parameters using least square loss function
println("Calculating parameters using least square loss function")
theta = X_train \ y_train
y_pred = X_test * theta
mae = compute_mae(y_pred, y_test)
println("MAE for least square loss function: $mae")


#Calculating parameters using huber loss function
println("Calculating parameters using huber loss function")
theta_huber = Variable(size(X_train, 2))
residual = X_train * theta_huber - y_train
loss = sum(custom_huber_loss(residual[i]) for i in 1:size(X_train, 1))
problem = minimize(loss)
solve!(problem, SCS.Optimizer, silent_solver = true)
y_pred = X_test * theta_huber.value
mae = compute_mae(y_pred, y_test)
println("MAE for huber loss function: $mae")

# Calculating parameters using hinge loss function
println("Calculating parameters using hinge loss function")
theta_hinge = Variable(size(X_train, 2))
residual = X_train * theta_hinge - y_train
loss = sum(custom_hinge_loss(residual[i]) for i in 1:size(X_train, 1))
problem = minimize(loss)
solve!(problem, SCS.Optimizer, silent_solver = true)
y_pred = X_test * theta_hinge.value
mae = compute_mae(y_pred, y_test)
println("MAE for hinge loss function: $mae")


