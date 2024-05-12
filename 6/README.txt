Supervised Learning

Works for both classification and regression

Foundation of Random Forests

Attractive because of interpretability

Decision Tree works by:

Split based on set impurity criteria
Stopping criteria
Source: Scikit-Learn

Some advantages of decision trees are:

Simple to understand and to interpret. Trees can be visualised.
Requires little data preparation.
Able to handle both numerical and categorical data.
Possible to validate a model using statistical tests.
Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
The disadvantages of decision trees include:

Overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
Decision trees can be unstable. Mitigant: Use decision trees within an ensemble.
Cannot guarantee to return the globally optimal decision tree. Mitigant: Training multiple trees in an ensemble learner
Decision tree learners create biased trees if some classes dominate. Recommendation: Balance the dataset prior to fitting
Questions:
What is a decision tree?

Where can you apply decision tree to? numerical problems or categorical problems?

Decision tree is also know by what other name?

How does a decision tree work?

Decision Tree is a foundation of what machine learning algorithm

List and explain 3 advantages of decision tree

List and explain 3 disadvantages of decision tree


Tree algorithms: ID3, C4.5, C5.0 and CART
ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.
C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. These accuracy of each rule is then evaluated to determine the order in which they should be applied. Pruning is done by removing a rule’s precondition if the accuracy of the rule improves without it.
C5.0 is Quinlan’s latest version release under a proprietary license. It uses less memory and builds smaller rulesets than C4.5 while being more accurate.
CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.
CHAID (Chi-squared Automatic Interaction Detector). by Gordon Kass. Performs multi-level splits when computing classification trees. Non-parametric. Does not require the data to be normally distributed.
scikit-learn uses an optimised version of the CART algorithm.

Gini Impurity
scikit-learn default

Gini Impurity

A measure of purity / variability of categorical data

As a side note on the difference between Gini Impurity and Gini Coefficient

No, despite their names they are not equivalent or even that similar.
Gini impurity is a measure of misclassification, which applies in a multiclass classifier context.
Gini coefficient applies to binary classification and requires a classifier that can in some way rank examples according to the likelihood of being in a positive class.
Both could be applied in some cases, but they are different measures for different things. Impurity is what is commonly used in decision trees.
Developed by Corrado Gini in 1912

Key Points:

A pure node (homogeneous contents or samples with the same class) will have a Gini coefficient of zero
As the variation increases (heterogeneneous classes or increase diversity), Gini coefficient increases and approaches 1.
Gini=1−∑jrp2j
p is the probability (often based on the frequency table)


Entropy
Wikipedia

The entropy can explicitly be written as

H(X)=∑i=1nP(xi)I(xi)=−∑i=1nP(xi)logbP(xi),
where b is the base of the logarithm used. Common values of b are 2, Euler's number e, and 10

Which should I use?
Sebastian Raschka

They tend to generate similar tree
Gini tends to be faster to compute


Information Gain
Expected reduction in entropy caused by splitting

Keep splitting until you obtain a as close to homogeneous class as possible




Tips on practical use
Decision trees tend to overfit on data with a large number of features. Check ratio of samples to number of features

Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand

Visualise your tree as you are training by using the export function. Use max_depth=3 as an initial tree depth.

Use max_depth to control the size of the tree to prevent overfitting.

Tune min_samples_split or min_samples_leaf to control the number of samples at a leaf node.

Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant.

By sampling an equal number of samples from each class

By normalizing the sum of the sample weights (sample_weight) for each class to the same value.






