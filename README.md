# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODETECH IT SOLUTIONS

NAME: VADLAMUDI PRASANNA

INTERN ID: CT08DF213

DOMAIN: MACHINE LEARNING

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

##This project demonstrates a basic sentiment analysis model using TF-IDF vectorization and Logistic Regression to classify customer reviews as positive or negative. Sentiment analysis is a common natural language processing (NLP) task that determines the emotional tone behind text, widely used in analyzing product reviews, feedback, and social media content. The program is implemented in Python using key libraries such as pandas for data manipulation, nltk for natural language processing, scikit-learn for machine learning, and matplotlib and seaborn for visualization. The dataset used in this project consists of a small set of manually created reviews labeled with sentiment (1 for positive, 0 for negative). Each review reflects a real-world customer experience, ranging from enthusiastic praise to strong dissatisfaction.

Before feeding the text into a machine learning model, preprocessing is essential to clean and standardize the data. First, all text is converted to lowercase to ensure consistency. Next, regular expressions are used to remove punctuation, numbers, and special characters, keeping only alphabetic words. Tokenization splits the text into individual words, and stopwords—common words like “the,” “is,” and “and” that do not carry sentiment—are removed using NLTK’s stopword list. This reduces noise and helps focus on meaningful terms. The cleaned reviews are stored in a new column and are ready for feature extraction.

For converting the text into numerical data, the TF-IDF (Term Frequency–Inverse Document Frequency) vectorizer is used. This method assigns a weight to each word based on how often it appears in a review (term frequency) and how unique it is across all reviews (inverse document frequency). Words that are common across many reviews receive lower weights, while words unique to specific reviews get higher weights. This results in a matrix where each review is represented by a vector of TF-IDF scores, making it suitable for training a machine learning model.

The dataset is then split into training and test sets, with 80% of the data used to train the model and 20% used to evaluate it. A Logistic Regression model is trained on the TF-IDF vectors. Logistic Regression is a linear classifier that predicts probabilities and is widely used for binary classification tasks like sentiment analysis. It is simple, interpretable, and effective for linearly separable data.

After training, the model is evaluated using accuracy, a classification report, and a confusion matrix. Accuracy measures the overall percentage of correct predictions. The classification report provides precision, recall, and F1-score, giving deeper insight into how well the model handles each class. The confusion matrix, visualized using Seaborn, shows the counts of true and false positives and negatives, helping to understand the types of errors the model makes.

In summary, this project provides a clear example of building a sentiment analysis system from scratch using traditional machine learning techniques. By applying text preprocessing, TF-IDF vectorization, and Logistic Regression, it demonstrates a full NLP pipeline suitable for small datasets and beginner-level classification tasks.

OUTPUT:
<img width="1612" height="665" alt="Image" src="https://github.com/user-attachments/assets/042de9b3-dfb5-4237-860a-79c263d276db" />
