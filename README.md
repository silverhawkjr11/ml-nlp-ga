# sentiment analysis and trying to implement text generation

# 1. Sentiment Analysis
## 1.1. Data
the data is from an amazon product reviews dataset that we took only 5000 reviews for the sake of easier training and testing. The data is in the form of a csv file with two columns: review and sentiment. The sentiment is either positive or negative.
## 1.2. Preprocessing
The data is preprocessed by removing stopwords,e punctuation, and converting the text to lowercase. The text is then tokenized and converted to sequences.
## 1.3. Model
we tried trainig the model in different ways:
- using a simple neural network
- using naive bayes
- using k nearest neighbors
- using a decision tree

summary of the results:
the nural network was the best when it comes to testing and it picks up on different sentiments , all the other models were quite good too apart from naive bayes was absolutely horrible.

# 2. Text Generation
## lemma: 
when rnns do the recurrence step they are prone to getting stuck in a local minimum so the idea is to use genetic algorithm to get out of the local minimum and find the global minimum.

## results:
there's potential but the dataset isnt vast enough to get a good result.