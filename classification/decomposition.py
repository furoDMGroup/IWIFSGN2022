import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import check_is_fitted


class OneVsRestClassifierForRandomBinaryClassifier(OneVsRestClassifier):
    def predict_proba(self, X):
        """Probability estimates.

                The returned estimates for all classes are ordered by label of classes.

                Note that in the multilabel case, each sample can have any number of
                labels. This returns the marginal probability that the given sample has
                the label in question. For example, it is entirely consistent that two
                labels both have a 90% probability of applying to a given sample.

                In the single label multiclass case, the rows of the returned matrix
                sum to 1.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)

                Returns
                -------
                T : (sparse) array-like of shape (n_samples, n_classes)
                    Returns the probability of the sample for each class in the model,
                    where classes are ordered as they are in `self.classes_`.
                """
        check_is_fitted(self)
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
            # If all classes have probability equal 0, then normalization to 1
            # gives vector of nans (because of division by zero).
            # Here we use an assumption that in such case
            # each class have equal probability
            Y[np.argwhere(np.isnan(Y))] = (1 / Y.shape[1])
        return Y
