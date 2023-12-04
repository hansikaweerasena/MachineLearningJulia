import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Define the proximal operator function
def proximal_operator(theta, lambda_, gamma):
    norm_theta = np.linalg.norm(theta, 2)
    if norm_theta != 0:
        return np.maximum(0, 1 - lambda_ * gamma / norm_theta) * theta
    else:
        return np.zeros_like(theta)

# Define the stochastic proximal subgradient descent function with evaluation
def stochastic_proximal_subgradient(X_train, Y_train, X_test, Y_test, k, lambda_, max_iter, eval_interval, gamma_t):
    n, d = X_train.shape  # Number of instances (n) and features (d)
    Theta = np.zeros((d, k))  # Initialize parameters for each class
    accuracies = []  # List to store accuracy for plotting
    
    for t in range(1, max_iter + 1):
        # Sample a random instance
        i = np.random.randint(0, n)
        xi, yi = X_train[i], Y_train[i]
        
        # Compute the subgradient for the hinge loss
        margins = xi @ Theta
        margins[yi] -= 1
        c = np.argmax(margins)
        g = np.zeros((d, k))
        if c != yi:
            g[:, c] = xi
            g[:, yi] = -xi
        
        # Update Theta
        Theta -= gamma_t / t * g
        for j in range(k):
            Theta[:, j] = proximal_operator(Theta[:, j], lambda_, gamma_t / t)
        
        # Evaluate on the test set every 'eval_interval' iterations
        if t % eval_interval == 0:
            Y_pred_test = predict(X_test, Theta)
            accuracy = accuracy_score(Y_test, Y_pred_test)
            accuracies.append(accuracy)
            print(f"Iteration {t}: Test accuracy = {accuracy}")
    
    return Theta, accuracies

# Predict function
def predict(X, Theta):
    scores = X @ Theta
    return np.argmax(scores, axis=1)

# Fetch the dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Extract features and labels using term frequencies
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test = vectorizer.transform(newsgroups_test.data).toarray()

# Convert labels to numerical values
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(newsgroups_train.target)
Y_test = label_encoder.transform(newsgroups_test.target)


# Parameters
k = np.unique(Y_train).size  # The number of unique labels
lambdas_ = [10]  # Regularization parameter
max_iter = 1000000  # Number of iterations for the SGD algorithm
eval_interval = 1000  # Evaluate every 1000 iterations

for lambda_ in lambdas_:
    # Run the algorithm
    Theta, accuracies = stochastic_proximal_subgradient(X_train, Y_train, X_test, Y_test, k, lambda_, max_iter, eval_interval, gamma_t=1.0)

    zero_coefficient_indices = np.all(Theta == 0, axis=1)  # Rows where all coefficients are zero
    feature_names = vectorizer.get_feature_names_out()  # Get the feature names (words)
    ignored_words = feature_names[zero_coefficient_indices]  #

    print(f"Number of ignored words: {ignored_words.size}")
    print(f"Ignored words: {ignored_words}")

    file_path = './sgd_news_all_min' + str(lambda_) + '.txt'
    with open(file_path, 'w') as file:
        for element in ignored_words:
            file.write(str(element) + '\n')

    # Plot the accuracies
    plt.plot(range(eval_interval, max_iter+1, eval_interval), accuracies)
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy for lambda =' + str(lambda_))

    plt.savefig('sgd_news_all_min' + str(lambda_) + '.png' )

    plt.close()