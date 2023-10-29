import numpy as np
class NaiveBayes():
    def fit(self, X, y):
        """
        getting some basic values
        :param X: training data
        :param y: training label
        :return:mean, variance of each variable based on each class and the prior of each class
        """
        self.X = X
        self.y = y
        # get a unique class type list
        self.classes = np.unique(y)
        self.parameters = {}
        for i, c in enumerate(self.classes):
            variable_c = X[np.where(y == c)]
            variable_c = np.array(variable_c)
            # caculating the mean
            variable_c_mean = np.mean(variable_c, axis=0, keepdims=True)
            # caculating the variance
            variable_c_sd = np.sqrt(np.var(variable_c, axis=0, keepdims=True))
            # caculating the possibility of each class
            variable_c_prior =  variable_c.shape[0] / X.shape[0]
            parameters = {"mean": variable_c_mean, "sd": variable_c_sd, "prior":variable_c_prior}
            self.parameters["class" + str(c)] = parameters

    def _pdf(self, X, y):
        """
        using Gaussian probability density function to calculate posterior
        :param X: variable
        :param y: class label
        :return: posterior value calculated by log
        """
        eps = 1e-5
        mean = self.parameters["class" + str(y)]["mean"]
        sd = self.parameters["class" + str(y)]["sd"]
        n = np.exp(-(X - mean) ** 2 / (2 * sd**2 + eps))
        d = np.sqrt(2 * np.pi )* sd + eps
        result = np.sum(np.log(n / d), axis=1, keepdims=True)
        return result.T

    def predict(self, X):
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters["class" + str(y)]["prior"])
            posterior = self._pdf(X, y)
            prediction = prior + posterior
            output.append(prediction)
        output = np.reshape(output, (self.classes.shape[0], X.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction

