---
template: bookdown-article.template
link-citations: true
citation-style: ieee-with-url.csl
bibliography: refs.bib
---

# Machine Learning Engineer Nanodegree
## Capstone Proposal
Yoong Kang Lim
August 1, 2018

## Proposal

The proposed project is the Jigsaw Toxic Comment Classification challenge on Kaggle [@jigsaw].

### Domain Background

There is a lot of discourse on the internet, and it is one of the things that keep people engaged on the internet. Comments can be found on various content such as news articles, blog posts, and forums. However, there can be instances of abuse and harrassment on these platforms. The threat of this may lead to people becoming less inclined to express their opinions on the internet [@duggan2014online].

Manual moderation of comments -- one done by a human being -- often works when the number of comments is small. Another approach is "crowd sourcing", which can be seen in platforms like Reddit which have upvotes and downvotes, where downvoted comments are hidden or deleted. However, this has some problems. With more and more people engaging in discussions on the internet, this can be too much for human beings to manage. As a result, more work is being done to automatically detect and classify toxic comments using machine learning [@WulczynTD16].

This project is an attempt to improve upon such machine learning models, particularly those provided by the Perspective API [@perspectiveapi]. While these models are useful, they are not perfect and are subject to several flaws. One of the flaws is that an intentional misspelling can affect the toxicity score significantly [@hosseini2017]. It also does not allow users to find the "type" of toxicity they are interested in. For example, some platforms may allow profanity but a binary classification system will mark a comment with profanity as a toxic comment. On the other hand, a comment without profanity may look civil on the surface, but is just as harmful, (e.g "I feel sorry for your children, as they are doomed to live a life raised by a mother with clearly deficient mental capacity").


### Problem Statement

The goal of the project is to predict the probability that a comment falls under a set of "toxicity" classes. The classes are `toxic`, `severe_toxic`, `obscene`, `threat`, and `identity_hate`. The model should tell us the probability that it falls under each class as a value between 0.0 and 1.0. For example it can be 0.6 toxic and 0.2 threat, etc.


### Datasets and Inputs

The datasets are provided by Jigsaw on the Kaggle competition website [@jigsaw].

The input is in CSV format with the following columns:

```
"id",
"comment_text",
"toxic",
"severe_toxic",
"obscene",
"threat",
"insult",
"identity_hate"
```

The dimensionality of the input is 160k rows with 8 columns.

Jigsaw and Kaggle have also provided the test set with dimensionality 153k rows with 2 columns (where the columns are `id` and the `comment_text`).

As this is a completed competition, they have also provided a test set with labels, so that submissions are unnecessary to find out the score.


### Solution Statement

The output will be in the format expected by Kaggle, which is a CSV file with the following columns:

```
"id",
"toxic",
"severe_toxic",
"obscene",
"threat",
"insult",
"identity_hate"
```

Note that this is the same as the training input, except without `comment_text`.

Apart from the column `id`, each column should contain the model's prediction that it falls under this class. This will be a value between 0.0 and 1.0, where the higher value means more likely. Thus, this is a multi output regression problem.

The solution is quantifiable by taking the difference between these probabilities with the ground truth. This score will be evaluated under the metric that will be described later in this proposal.

This will be provided as Jupyter notebook(s), so that it is replicable.


### Benchmark Model

The benchmark model I will be using to compare my model with is the top voted kernel on Kaggle for this competition:

* NB-SVM strong baseline (https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline)


### Evaluation Metrics

The evaluation metric will be the same as the Kaggle competition's metric, which is mean column-wise ROC (receiver operating characteristic) AUC (area-under-curve). In other words, the score is the average of the individual AUCs of each predicted column.


### Project Design

These are the phases that I will use to find a solution for the given problem:

**Exploratory data analysis**

I will look at the data to look at what the training input looks like. I will identify things that require some data cleaning or feature engineering. For example, it was not specified if all the comments are in English -- and if there are Japanese comments in an encoding like Shift-JIS, I will need to deal with this.

I will also collect some simple statistics, like obtaining the most common words associated with a certain class, and to see if there is any overlap. This is to see which words are important for analysis, which can help inform what feature engineering I do.

**Preprocessing and feature engineering**

It is likely that this project will need feature engineering. Comments are likely not of the same length, so I will need to find a way to make that matter less. For example, a naive way would be to simply repeat the comment until it is the same size as the longest comment. I will take a small sample to see if this improves predictions.

Some other things I will explore is to use TF-IDF, as well as ngrams. Another thing I might do is to preprocess the comments to fix simple spelling mistakes.

The EDA phase should inform what feature engineering might be required.

**Model and hyperparameter selection**

Kaggle has given us a training set and a test set. I will break the training set into training + validation sets. This is to tune hyperparameters.

If needed, I will split the training set into three (training + validation + test), which will avoid accidentally overfitting to the test set (e.g by submitting and refining the model).

As we seem to have enough data, I might skip cross validation techniques, but I am not ruling it out right now.

In machine learning, it is best to select a validation set that is as similar as possible to the test set, so there will also be a step to calibrate my validation set. For example, the dataset may be unbalanced, so it may be beneficial to maintain that same balance in the validation and test set.

At this stage, I plan look at two types of models. The first one I will look at will be NB-SVM (as described in the Benchmark Model section) with my own improvements. As it takes a shorter time to train NB-SVM models, this will be what I might use to see the affect of any feature engineering I do.

The second type of model I will explore is a deep neural network. There is a wide range of things I can try, but these two will be the first things I will try:

* A neural network with 1-3 hidden fully connected layers
* A bidirectional LSTM with 1-3 connected layers.

More work will need to be done to determine how many layers to choose.

**Model refinement**

I will make further refinements to the model. For example, for deep neural networks I may experiment with dropout, early stopping, and various activation functions for hidden layers (relu, tanh, etc.).

I might also experiment with the architecture. One path that I might go down is to replace LSTM with GRU units, as it has been shown empirically to produce good results.

**Report**

I will then describe all my steps in a report and provide Jupyter notebook(s) that document the steps I took.


## References