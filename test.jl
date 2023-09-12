using LinearAlgebra

# genrating random matrix and vector
Phi = randn(40, 10)
psi = randn(40)

# solving the least square
theta_star = Phi\psi 

# calculate associated loss
loss = norm(Phi*theta_star - psi)^2
println("Loss : ", loss)

# checks for the expected inequality
for i in 1:10
    delta = randn(10)
    loss_delta = norm(Phi*(theta_star + delta) - psi)^2
    println("Loss with delta: ", loss_delta)
    if loss_delta > loss
        println("Loss with delta is greater than loss, inequality holds")
    else
        println("Loss with delta is less than loss, inequality does not hold")
    end        
end

