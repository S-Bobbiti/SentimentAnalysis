# SentimentAnalysis
This repository consists of my assignment work that was done in course ECE 657: Tools of Intelligent System Design. 

The IMDB large movie review dataset consists of many positive and negative reviews on movies and is used for sentiment analysis. It is sourced from Stanford database [1]. For classifying the reviews into either positive or negative, we can use several models ranging from logistic regression, naive bayes to CNNs, RNNs [2].

I choose to train a simple CNN and an LSTM model and compared their perofrmance.

Comments:
- The CNN model performs the best and has a very good train, validation, and test accuracy.
- LSTM takes a long time to train and still has a lower performance. Overall, it exhibits low accuracy and high loss.

Learning curves:
- CNN: It was observed that the generalization gap between the training and validation curves is less, and the curves are smooth. Indicating an optimal performance.
- LSTM: The learning curves of LSTM model shows random fluctuations and doesn't fit well. 

And hence for the above reasons I choose CNN model to be the final model for sentiment analysis.

References:

[1]. http://ai.stanford.edu/~amaas/data/sentiment/

[2]. https://www.datarobot.com/blog/using-machine-learning-for-sentiment-analysis-a-deep-dive/
