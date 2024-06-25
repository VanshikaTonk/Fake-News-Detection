# Fake-News-Detection

PROJECT OVERVIEW:
In today's digital age, the spread of fake news has become a significant concern. This project aims to develop a machine learning-based system to detect fake news articles and curb their spread. The system uses natural language processing (NLP) techniques, sentiment analysis, named entity recognition (NER), topic modeling, and data mining to identify the authenticity of news articles.

TECHNOLOGIES USED:
Natural Language Processing (NLP): for text analysis and feature extraction
Tokenization: breaking down text into individual words or tokens
Stopword removal: removing common words like "the", "and", etc. that do not add much value to the text
Lemmatization: reducing words to their base form (e.g. "running" becomes "run")
Vectorization: converting text into numerical vectors that can be processed by machine learning algorithms
Machine Learning (ML): for building and training the fake news detection model
Supervised learning: using labeled data to train the model
PassiveAggressiveClassifier: a machine learning algorithm used for building the fake news detection model
Sentiment Analysis: for analyzing the sentiment of news articles
Positive/Negative sentiment: identifying the sentiment of news articles as positive, negative, or neutral
Named Entity Recognition (NER): for identifying entities mentioned in news articles
Person, Organization, Location: identifying entities such as people, organizations, and locations mentioned in news articles
Topic Modeling: for identifying topics and themes in news articles
Latent Dirichlet Allocation (LDA): a topic modeling technique used to identify topics and themes in news articles
Data Mining: for extracting insights from large datasets
Data preprocessing: cleaning and preprocessing the dataset for analysis
Data visualization: visualizing the results of the analysis using plots and charts

DOMAIN:
Social: focusing on social media platforms and online news sources

DIFFICULTY LEVEL:
Medium: requiring a good understanding of NLP, ML, and data analysis concepts

PROBLEM STATEMENT:
The spread of fake news has become a significant concern in today's connected world. Fake news can cause problems to individuals, organizations, and society as a whole. This project aims to develop a system to identify the authenticity of news articles and curb the spread of fake news.

SOLUTION APPROACH:
The system uses a combination of NLP techniques, sentiment analysis, NER, topic modeling, and data mining to analyze news articles and predict whether they are fake or real. The system is built using Python and the following libraries:

Pandas: for data manipulation and analysis
NumPy: for numerical computations
Scikit-learn: for machine learning and model building
TfidfVectorizer: for feature extraction and vectorization
PassiveAggressiveClassifier: for building the fake news detection model
Dataset
The dataset used for this project is News.csv, which contains a collection of news articles labeled as fake or real. The dataset is divided into training and testing sets, with 80% of the data used for training and 20% used for testing.

MODEL EVALUATION:
The system is evaluated using metrics such as:

Accuracy: the proportion of correctly classified news articles
Precision: the proportion of true positives (correctly classified fake news articles) among all positive predictions
Recall: the proportion of true positives among all actual fake news articles
F1-score: the harmonic mean of precision and recall
Results
The system achieves an accuracy of 85%, with a precision of 80%, recall of 90%, and F1-score of 85%.

FUTURE WORK:
Future work includes:

Improving the accuracy of the system: by experimenting with different machine learning algorithms and NLP techniques
Expanding the dataset: by collecting more news articles and labeling them as fake or real
Integrating with social media platforms: by integrating the fake news detection system with social media platforms to detect and flag fake news in real-time
Contributing
Contributions to this project are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

LICENSE:
This project is licensed under the MIT License.

ACKNOWLEDGEMENTS:
This project was completed as part of an internship at Turing Technologies. I would like to thank Turing Technologies for provifing me this wonderful project which helped me in gaining valueable insights in the feild of Machine Learning.


