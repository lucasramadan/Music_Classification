# Annotations by Lucas Ramadan

# some crazy multi-inheritence, I looked up ABCMeta, but I don't understand it
class BaseNB(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """Abstract base class for naive Bayes estimators"""

    # LR: just setting the joint_log_likelihood (jll) function for later classes
    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X
        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape [n_classes, n_samples].
        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        """

    # LR: this function gets the jll and then just returns the most likely label
    # which is the max for each COLUMN of jll, ie a single sample, for each

    # sort of looks like this for two class system:
    """
           n_samples
     c=0 [[         ]
     c=1  [         ]]

    """

    # LR: self.classes_ comes from labelbin object below, but is just a label
    # since it calls fit_transform on y, which are the labels

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    # LR: logsumexp is an imported function, but the key here is that theyre
    # calculating the log probability for each sample, for each class
    # which they will then use later (ie to take the most likely class)

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    # LR: this is a little back to front, but it's basically just undoing
    # the log proba from above, to get back to non-log form

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        return np.exp(self.predict_log_proba(X))

# LR: Now we're in the real meat of the objects
# this BaseDiscreteNB inherits from the BaseNB
# and is going to later be used for the actual MultinomialNB

class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data
    Any estimator based on this class should provide:
    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

    # LR: this function is sort of intuitive, it's basically checking to see
    # if there are already calculated class_priors, which need to be logged
    # it raises an error if there are unequal priors and classes, because
    # you need a prior probability for each class

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)

        # LR: this is a little tricky, but basically if the NB object has
        # the boolean flag .fit_prior set to True, then it's going to calculate
        # the log priors as the count of a given class divided by the total
        # count of all classes. Since we're taking the log, it ends up being
        # subtraction, because of the identity: log(a/b) = log(a) - log(b)


        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_)
                                     - np.log(self.class_count_.sum()))

        # LR: else means that the user did NOT want to fit_prior
        # so they just initialize as a uniform prior

        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    # LR: this is basically the same as the fit method, but does it bit by bit
    # so that you can train your model on chunks of your data, such as for
    # online data streaming in, or what-have-you. Ignore for our purposes.

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        classes : array-like, shape = [n_classes]
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)
        elif n_features != self.coef_.shape[1]:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (n_features, self.coef_.shape[-1]))

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_samples, n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=class_prior)
        return self


    # LR: just a regular fit method, with you being allowed to specify
    # sample weights, just like we did for AdaBoosting. I suspect you could
    # boost your naive bayes model, which is why this is included.

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
            Returns self.
        """

        # LR: seems like this is just a check to see if X, y are in sparse
        # matrix representation, so that they can use it later on
        # assign n_features based off of the shape of X (makes sense)

        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        # LR: initialize LabelBinarizer object, which is where they get the
        # labels for y from, as well as the classes that they use earlier for
        # the predict methods. I think it's just there to convert string labels
        # like ('good', 'bad') into binary (1, 0) versions for math later on

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)

        # LR: basically if the user wanted to weight the samples when they
        # initialized the naive bayes object, then we need to update Y ?

        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        # LR: comes from initialization of the NB object, default is None

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas

        # LR: based off of the binarize function, im guessing this just gets
        # the number of classes possible based off of the binarization

        # LR: so for example if there are three classes, then the binarizer
        # would return for a given sample, something like: [1, 0, 0] meaning the
        # given sample is class 1 out of 3.

        # LR: they then just initialize class_count and feature_count as zeros
        # to be filled in later

        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self._count(X, Y)
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=class_prior)
        return self

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    # LR: little weird, but as far as I can tell from looking this up
    # it looks like this just makes coef_ and intercept_ equivalent to class
    # attributes, such that you can get them from the object after fitting?

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)

# LR: finally, our guy --- it's going to inherit from BaseDiscreteNB which we
# learned inherits from BaseNB and so on up the chain

class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models
    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.
    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.
    intercept_ : property
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.
    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.
    coef_ : property
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.
    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """

    # LR: our usual constructor, allowing user to specific desired attributes

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    # LR: clever way to check the format of the X matrix, basically raises an
    # error if the counts are negative, being flexible for sparce and dense
    # representations of the data in X

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        # LR: as they did above, now they're filling in feature_count which is
        # basically just a matrix of (n_classes, n_features) shape
        # for which a column is the sum of feature x_i for each class
        # sort of like S_y,i from our NB classifier

        self.feature_count_ += safe_sparse_dot(Y.T, X)

        # LR: this is then just the count of all classes, since we know
        # that Y is a binarized matrix of classes. IE something like:

        """
        [[1, 0, 0]
         [0, 0, 1]
         [0, 1, 0]]
        """

        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        # LR: this is just like our NB, where they're first calculating the
        # numberator (smoothed_fc) and then the denomenator (smoothed_cc)
        # again keeping in mind that log(a/b) = log(a) - log(b)

        self.feature_log_prob_ = (np.log(smoothed_fc)
                                  - np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        # LR: they just have a sparse matrix check above
        # and then they return the dot of X and the log probas
        # since log(p()^2) = 2log(p) ... just like in our lab

        return (safe_sparse_dot(X, self.feature_log_prob_.T)
                + self.class_log_prior_)

    # LR: to follow this all the way through, you then have to go back up
    # into the BaseDiscreteNB class, into the fit method and notice that
    # they just return self at the end of the fit call, which I suppose
    # is just the child object itself, so in this case the MultinomialNB obj
