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

## Output:

The model has been trained for 20 epochs and accuracy vs number of epochs and loss vs the number of epochs curves are plotted for both training and validation sets.

![Figure_2](https://user-images.githubusercontent.com/106268058/228070392-1210792e-ce5f-4554-a07f-93aba2ab4992.png)


![Figure_1](https://user-images.githubusercontent.com/106268058/228070414-d19f34a0-bc1f-4d56-b931-bc1edf216829.png)


Training loss at 20th epoch: 0.055 
Training accuracy at 20th epoch: 0.983 

Validation accuracy at 20th epoch: 0.904 
Validation loss at 20th epoch: 0.292

Model results for test dataset:

Test accuracy: 87.59%
Test loss: 37.88%

References:

[1]. http://ai.stanford.edu/~amaas/data/sentiment/

[2]. https://www.datarobot.com/blog/using-machine-learning-for-sentiment-analysis-a-deep-dive/
