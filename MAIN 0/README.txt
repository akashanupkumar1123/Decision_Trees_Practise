In this exercise, you will implement a decision tree from scratch and apply it to the task of classifying whether a mushroom is edible or poisonous.



2 - Problem Statement
Suppose you are starting a company that grows and sells wild mushrooms.

Since not all mushrooms are edible, you'd like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
You have some existing data that you can use for this task.
Can you use the data to help you identify which mushrooms can be sold safely?


3 - Dataset
You will start by loading the dataset for this task. The dataset you have collected is as follows:

Cap Color	Stalk Shape	Solitary	Edible
Brown	Tapering	Yes	1
Brown	Enlarging	Yes	1
Brown	Enlarging	No	0
Brown	Enlarging	No	0
Brown	Tapering	Yes	1
Red	Tapering	Yes	0
Red	Enlarging	No	0
Brown	Enlarging	Yes	1
Red	Tapering	No	1
Brown	Enlarging	No	0
You have 10 examples of mushrooms. For each example, you have
Three features
Cap Color (Brown or Red),
Stalk Shape (Tapering or Enlarging), and
Solitary (Yes or No)
Label
Edible (1 indicating yes or 0 indicating poisonous)

3.1 One hot encoded dataset
For ease of implementation, we have one-hot encoded the features (turned them into 0 or 1 valued features)

Brown Cap	Tapering Stalk Shape	Solitary	Edible
1	1	1	1
1	0	1	1
1	0	0	0
1	0	0	0
1	1	1	1
0	1	1	0
0	0	0	0
1	0	1	1
0	1	0	1
1	0	0	0
Therefore,

X_train contains three features for each example

Brown Color (A value of 1 indicates "Brown" cap color and 0 indicates "Red" cap color)
Tapering Shape (A value of 1 indicates "Tapering Stalk Shape" and 0 indicates "Enlarging" stalk shape)
Solitary (A value of 1 indicates "Yes" and 0 indicates "No")
y_train is whether the mushroom is edible

y = 1 indicates edible
y = 0 indicates poisonous






View the variables
Let's get more familiar with your dataset.

A good place to start is to just print out each variable and see what it contains.
The code below prints the first few elements of X_train and the type of the variable.




Check the dimensions of your variables
Another useful way to get familiar with your data is to view its dimensions.

Please print the shape of X_train and y_train and see how many training examples you have in your dataset.




4 - Decision Tree Refresher
In this practice lab, you will build a decision tree based on the dataset provided.

Recall that the steps for building a decision tree are as follows:
Start with all examples at the root node
Calculate information gain for splitting on all possible features, and pick the one with the highest information gain
Split dataset according to the selected feature, and create left and right branches of the tree
Keep repeating splitting process until stopping criteria is met
In this lab, you'll implement the following functions, which will let you split a node into left and right branches using the feature with the highest information gain

Calculate the entropy at a node
Split the dataset at a node into left and right branches based on a given feature
Calculate the information gain from splitting on a given feature
Choose the feature that maximizes information gain
We'll then use the helper functions you've implemented to build a decision tree by repeating the splitting process until the stopping criteria is met

For this lab, the stopping criteria we've chosen is setting a maximum depth of 2

4.1 Calculate entropy
First, you'll write a helper function called compute_entropy that computes the entropy (measure of impurity) at a node.

The function takes in a numpy array (y) that indicates whether the examples in that node are edible (1) or poisonous(0)
Complete the compute_entropy() function below to:

Compute 𝑝1, which is the fraction of examples that are edible (i.e. have value = 1 in y)
The entropy is then calculated as
𝐻(𝑝1)=−𝑝1log2(𝑝1)−(1−𝑝1)log2(1−𝑝1)
Note
The log is calculated with base 2
For implementation purposes, 0log2(0)=0. That is, if p_1 = 0 or p_1 = 1, set the entropy to 0
Make sure to check that the data at a node is not empty (i.e. len(y) != 0). Return 0 if it is


4.2 Split dataset
Next, you'll write a helper function called split_dataset that takes in the data at a node and a feature to split on and splits it into left and right branches. Later in the lab, you'll implement code to calculate how good the split is.

The function takes in the training data, the list of indices of data points at that node, along with the feature to split on.
It splits the data and returns the subset of indices at the left and the right branch.
For example, say we're starting at the root node (so node_indices = [0,1,2,3,4,5,6,7,8,9]), and we chose to split on feature 0, which is whether or not the example has a brown cap.




For each index in node_indices
If the value of X at that index for that feature is 1, add the index to left_indices
If the value of X at that index for that feature is 0, add the index to right_indices



4.3 Calculate information gain
Next, you'll write a function called information_gain that takes in the training data, the indices at a node and a feature to split on and returns the information gain from the split.


Exercise 3
Please complete the compute_information_gain() function shown below to compute

Information Gain=𝐻(𝑝node1)−(𝑤left𝐻(𝑝left1)+𝑤right𝐻(𝑝right1))
 
where

𝐻(𝑝node1)  is entropy at the node
𝐻(𝑝left1)  and  𝐻(𝑝right1)  are the entropies at the left and the right branches resulting from the split
𝑤left  and  𝑤right  are the proportion of examples at the left and right branch, respectively
Note:

You can use the compute_entropy() function that you implemented above to calculate the entropy
We've provided some starter code that uses the split_dataset() function you implemented above to split the dataset





4.4 Get best split
Now let's write a function to get the best feature to split on by computing the information gain from each feature as we did above and returning the feature that gives the maximum information gain


Exercise 4
Please complete the get_best_split() function shown below.

The function takes in the training data, along with the indices of datapoint at that node
The output of the function is the feature that gives the maximum information gain
You can use the compute_information_gain() function to iterate through the features and calculate the information for each feature If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.






5 - Building the tree
In this section, we use the functions you implemented above to generate a decision tree by successively picking the best feature to split on until we reach the stopping criteria (maximum depth is 2).




















