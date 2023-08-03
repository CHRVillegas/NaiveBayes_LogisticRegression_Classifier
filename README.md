## Usage
For Naive Bayes in main.py be sure to set the required model to NaiveBayesClassifier() and to fit your desired data to the model. Then finally call predict to get accuracy scores.


model = NaiveBayesClassifier()
model.fit(X_[Setname], Y_[Setname])
accuracy(model.predict(X_[Setname]), Y_[Setname])


For Logistic Regression (Similarly to Naive Bayes) set the model and fit data to the model using the text and label numpy arrays. Then use predict to get an accuracy score for the model.


model = LogisticRegressionClassifier()
model.fit(X_[Setname], Y_[Setname])
accuracy(model.predict(X_[Setname]), Y_[Setname])


Unlike Naive Bayes the Logistic Regression Model has 3 parameters that can be modified within the classifier.py file to change how it operates. The three parameters that can be set include iterations, alpha, and mylambda. Iterations represents the number of times that the gradient descent algorithm runs, Alpha is the learning rate of the algorithm, and mylambda is the lambda used for L2 Regularization.
