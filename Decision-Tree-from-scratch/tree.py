import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """

    EPS = 0.0005
    probs = y.sum(axis=0) / y.shape[0]
    return np.sum(-probs * np.log(probs + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    
    probs = y.sum(axis=0) / y.shape[0]
    return 1 - np.sum(probs ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
        
    return (1 / y.shape[0]) * np.sum((y - np.mean(y)) ** 2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    return (1 / y.shape[0]) * np.sum(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
        self.ans = None
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        mask_left = X_subset[:, feature_index] < threshold
        X_left = X_subset[mask_left]
        y_left = y_subset[mask_left]

        mask_right = X_subset[:, feature_index] >= threshold
        X_right = X_subset[mask_right]
        y_right = y_subset[mask_right]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        mask_left = X_subset[:, feature_index] < threshold
        y_left = y_subset[mask_left]

        mask_right = X_subset[:, feature_index] >= threshold
        y_right = y_subset[mask_right]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        feature_index, threshold = None, None
        max_G = 0

        for j in range(0, X_subset.shape[1]):
            for t in X_subset[:, j]:
                y_left, y_right = self.make_split_only_y(j, t, X_subset, y_subset)

                L = y_left.shape[0]
                R = y_right.shape[0]
                Q = y_subset.shape[0]

                if L == 0 or R == 0:
                    continue

                G = self.criterion(y_subset) - (L / Q) * self.criterion(y_left) - (R / Q) * self.criterion(y_right)

                if G > max_G:
                    feature_index, threshold = j, t
                    max_G = G

        return feature_index, threshold

    def calc_ans(self, y):
        """
        Makes a prediction for objects in the leaf
        
        Parameters
        ----------
        y : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        pred : np.array of type float with shape (, n_classes) in classification, 
               integer in regression
        """

        pred = None
        if self.classification:
            pred = y.sum(axis=0) / y.shape[0]
        else:
            if self.criterion_name == 'variance':
                pred = np.mean(y)
            elif self.criterion_name == 'mad_median':
                pred = np.median(y)
        return pred
    
    def pure(self, y):
        """
        Checks if the impurity of objects is good
        
        Parameters
        ----------
        y : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        True or false
        """

        if self.classification:
            classes = one_hot_decode(y)
            return np.unique(classes).shape[0] == 1
        else:
            return np.unique(y).shape[0] == 1


    def make_tree(self, X_subset, y_subset, depth=1):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)

        if X_subset.shape[0] < self.min_samples_split or depth >= self.max_depth \
            or self.pure(y_subset) or threshold is None:
            leaf = Node(None, None)
            leaf.ans = self.calc_ans(y_subset)
            return leaf

        new_node = Node(feature_index, threshold)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node.left_child = self.make_tree(X_left, y_left, depth + 1)
        new_node.right_child = self.make_tree(X_right, y_right, depth + 1)

        self.depth = max(self.depth, depth)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
        return self
        
    
    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, ) in classification 
                   (n_objects, ) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        y_predicted = []
        for x in X:
            node = self.root
            while node.threshold != None:
                if x[node.feature_index] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            if self.classification:
                y_predicted.append(np.argmax(node.ans))
            else:
                y_predicted.append(node.ans)
        
        return np.array(y_predicted)
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = []
        for x in X:
            node = self.root
            while node.threshold != None:
                if x[node.feature_index] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            y_predicted_probs.append(node.ans)
        
        return np.array(y_predicted_probs)
