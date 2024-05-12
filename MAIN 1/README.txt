The decision tree is one of the most widely used machine learning algorithms due to its ease of interpretation. However, do you know how it works? In this post I am going to explain everything that you need about decision trees. To do this, we are going to create our own decision tree in Python from scratch



Understanding how a decision tree works
A decision tree consists of creating different rules by which we make the prediction. For example, let’s say we train an algorithm that predicts whether or not a person is obese based on their height and weight.


Imagine that we want to predict whether or not the person is obese. Based on the description of the dataset (available on Kaggle), people with an index of 4 or 5 are obese, so we could create a variable that reflects this:



In that case, a decision tree would tell us different rules, such as that if the person’s weight is greater than 100kg, it is most likely that the person is obese. However, that cut will not be precise: there will be people who weigh 100kg or more who are not obese. Thus, the decision tree continues to create more branches that generate new conditions to “refine” our predictions.el sitio web de Betcris Perú

As you can see, decision trees usually have sub-trees that serve to fine-tune the prediction of the previous node. This is so until we get to a node that does not split. This last node is known as a leaf node or leaf node. Let’s see a graphic example:

Besides,a decision trees can work for both regression problems and for classification problems. In fact, we will code a decision tree from scratch that can do both.

Now you know the bases of this algorithm, but surely you have doubts. How does the algorithm decide which variable to use as the first cutoff? How do you choose the values?


Impurity and cost functions of a decision tree
As in all algorithms, the cost function is the basis of the algorithm. In the case of decision trees, there are two main cost functions: the Gini index and entropy.

Any of the cost functions we can use are based on measuring impurity. Impurity refers to the fact that, when we make a cut, how likely is it that the target variable will be classified incorrectly.

In the example above, impurity will include the percentage of people that weight >=100 kg that are not obese and the percentage of people with weight<100 kg that are obese. Every time we make a split and the classification is not perfect, the split is impure.

However, this does not mean that all cuts are the same: sure that the cut in 100kg classifies better than if we make the split at 80kg. In fact, we can check it:



n short, the cost function of a decision tree seeks to find those cuts that minimize impurity. Now, let’s see what ways exist to calculate impurity:

Calculate impurity using the Gini index
The Gini index is the most widely used cost function in decision trees. This index calculates the amount of probability that a specific characteristic will be classified incorrectly when it is randomly selected.

This is an index that ranges from 0 (a pure cut) to 0.5 (a completely pure cut that divides the data equally). The Gini index is calculated as follows:

Gini=1–∑i=1n(Pi)2
Where Pi is the probability of having that class or value.

Let’s program the function, considering the input will be a Pandas series:


As we can see, the Gini index for the Gender variable is very close to 0.5. This indicates that the Gender variable is very impure, that is, the cutting results are not will both have equally the same proportion of incorrectly classified data.

Now that you know how the index works, let’s see how entropy works.

Calculate impurity with entropy
Entropy it is a way of measuring impurity or randomness in data points. Entropy is defined by the following formula:

E(S)=∑i=1c−pilog2pi
Unlike the Gini index, whose range goes from 0 to 0.5, the entropy range is different, since it goes from 0 to 1. In this way, values close to zero are less impure than those that approach 1.

Let’s see how entropy works by calculating it for the same example that we have done with the Gini index:




As we see, it gives us a value very close to 1, which denotes an impurity similar to that indicated by the Gini impurity, whose value is close to 0.5.

With this, you already know the two main methods that can be used in a decision tree to calculate impurity. Perfect, we already know how to decide if a cut is good or not, but… between which splits do we choose? Let’s see it!

How to choose the cuts for our decision tree
As we have seen, cuts are compared by impurity. Therefore, we are interested in comparing those cuts that generate less impurity. For this, Information Gain is used. This metric indicates the improvement when making different partitions and is usually used with entropy (it could also be used with the Gini index, although in that case it would not be called Informaiton Gain).

The calculation of the Information Gain will depend on whether it is a classification or regression decision tree. There would be two options:

InformationGainClassification=E(d)–∑|s||d|E(s)
InformationGainRegresion=Variance(d)–∑|s||d|Variance(s)


Knowing this, the steps that we need to follow in order to code a decision tree from scratch in Python are simple:

Calculate the Information Gain for all variables.
Choose the split that generates the highest Information Gain as a split.
Repeat the process until at least one of the conditions set by hyperparameters of the algorithm is not fulfilled.
However, we have a newly added difficulty, and it is, how do we choose which is the best split in the numerical variables? And if there is more than one categorical variable?

How to calculate the best split for a variable
To calculate the best split of a numeric variable, first, all possible values that the variable is taking must be obtained. Once we have the options, for each option we will calculate the Information Gain using as a filter if the value is less than that value. Obviously, the first possible data will be drop, because the split will include all values.

In case we have categorical variables, the idea is the same, only that in this case we will have to calculate the Information Gain for all possible combinations of that variable, excluding the option that includes all the options (since it would not be doing any split). This is quite computationally costly if we have a high number of categories, that decision tree algorithms usually only accept categorical variables with less than 20 categories.

So, once we have all the splits, we will stick with the split that generates the highest Information Gain.



Now that we know how to calculate the split of a variable, let’s see how to decide the best split.

How to choose the best split
As I have previously said, the best split will be the one that generates the highest Information Gain. To know which one is it, we simply have to calculate the Information Gain for each of the predictor variables of the model.



As we can see, the variable with the highest Information Gain is Weight. Therefore, it will be the variable that we use first to do the split. In addition, we also have the value on which the split must be performed: 103.

With this, we already have the first split, which would generate two dataframes. If we apply this recursively, we will end up creating the entire decision tree (coded in Python from scratch). Let’s do it!

How to train a decision tree in Python from scratch
Determining the depth of the tree
We already have all the ingredients to calculate our decision tree. Now, we must create a function that, given a mask, makes us a split.

In addition, we will include the different hyperparameters that a decision tree generally offers. Although we could include more, the most relevant are those that prevent the tree from growing too much, thus avoiding overfitting. These hyperparameters are as follows:

max_depth: maximum depth of the tree. If we set it to None, the tree will grow until all the leaves are pure or the hyperparameter min_samples_split has been reached.
min_samples_split: indicates the minimum number of observations a sheet must have to continue creating new nodes.
min_information_gain: the minimum amount the Information Gain must increase for the tree to continue growing.
With this in mind, let’s finish creating our decision tree from 0 in Python. To do this, we will:

Make sure that the conditions established by min_samples_split and max_depth are being fulfilled.
Make the split.
Ensure that min_information_gain if fulfilled.
Save the data of the split and repeat the process.
To do this, first of all, I will create three functions: one that, given some data, returns the best split with its corresponding information, another that, given some data and a split, makes the split and returns the prediction and finally, a function that given some data, makes a prediction.

Note: the prediction will only be given in the branches and basically consists of returning the mean of the data in the case of the regression or the mode in the case of the classification.


Training our decision tree in Python
Now that we have these three functions, we can, let’s train the decision tree that we just programmed in Python.

We ensure that both min_samples_split and max_depth are fulfilled.
If they are fulfilled, we get the best split and obtain the Information Gain. If any of the conditions are not fulfilled, we make the prediction.
We check that the Information Gain Comprobamos passes the minimum amount set by min_information_gain.
If the condition above is fulfilled, we make the split and save the decision. If it is not fulfilled, then we make the prediction.
We will do this process recursively, that is, the function will call itself. The result of the function will be the rules you follow to make the decision:


It is done! The decision tree we just coded in Python has created all the rules that it will use to make predictions.

Now, there would only be one thing left: convert those rules into concrete actions that the algorithm can use to classify new data. Let’s go for it!

Predict using our decision tree in Python
To make the prediction, we are going to take an observation and the decision tree. These decisions can be converted into real conditions by splitting them.

So, to make the prediction we are going to:

Break the decision into several chunks.
Check the type of decision that it is (numerical or categorical).
Considering the type of variable that it is, check the decision boundary. If the decision is fulfilled, return the result, if it is not, then continue with the decision..


So, we can try to classify all the data in our algorithm to see how well our decision tree has worked that we just programmed in Python:



The decision tree we just coded in Python is almost 85% accurate! As we can see, it seems that it has trained well, although perhaps the hyperparameters that we have chosen are not the best (that’s a topic for another post).

Finally, as we have coded our decision tree in Python to support different types of data and to be used for both regression and classification, we are going to test it with different use cases:

Decision tree prediction for regression
To do the regression, we are going to use the gapminder dataset, which has the information on the number of inhabitants, GDP per Capita of different countries for different years.

So, we are going to use the algorithm to predict the life expectancy of a country taking into account its GDP per Capita and its population:


Likewise, we can take advantage of this same Gapminder dataset to check how, if we pass a categorical variable with more levels than what is set by the max_categories parameter, it will return an error:



Of course, we could also use the same prediction function to make predictions with trees that use categorical variables. Although, as you might expect, in this case the predictions will be very poor due to oversimplification of the predictor variables used:










































































