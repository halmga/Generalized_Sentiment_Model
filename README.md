# Generalized_Sentiment_Model
## Summary
Below is the Proof of Concept that was submitted with the code as a project for a Natural Language Processing course (code still being uploaded).

## Abstract
Sentiment analysis has been at the forefront of research and business decisions
for many years. For this reason, we have
created a generalized model for labelling
the sentiment of reviews from different forums
and domains. To create this model,
we train identical models on two different
datasets with similar features. We then use
both datasets and the information gain of
their features to extract the most similar
features with high information gain. These
features are then used in the final generalized
model. Using our finished generalized
model, we then tested it again on
the Yelp and IMDB datasets and an unseen
Coursera dataset obtaining accuracy
scores of over 80% for all datasets.
## Introduction
Whether it be supervised, unsupervised, at
document-level, or at sentence level, sentiment
analysis has been studied from every direction
with the hopes of finding the most efficient method
to understand what sentiment a writer is trying to
convey and the strength of these sentiments when
they create a text (Feldman , 2013). In the domain
of natural language processing, sentiment analysis
is one of the hottest topics, generating thousands
of research papers, new articles, and spawning
multiple startup companies. The reason for this
field’s popularity is clear. For researchers, comprehending
how emotionally charged words and
different sentence structures are can better help
them understand people’s language use in different
contexts. This can be helpful in discovering
how the semantic orientation of words can change
based on context (Wilson et al, 2015) or be used
to identify connections between different online
communities (Liu et al., 2014). For businesses,understanding the sentiment of consumers towards
different products and services in the market can
be very profitable. With the explosion of social
media and different review forums on the internet,
it is increasingly easier for companies to get
feedback from their customers. However, with so
many different platforms for consumers to leave
reviews on, it can be difficult for companies consolidate
and process these different reviews in a
uniform fashion to get an overarching picture of
what the population’s feelings are regarding their
product. This is one of the reasons why we have
explored building a model based on more generalized
features to be able to predict sentiment across
different platforms and product domains.
## Goal
This project’s aim is to develop a model that can
predict sentiment analysis of reviews over a diverse
range of forums and domains. This idea
pulls inspiration from a recent paper by (Agrawal
& Awekar, 2018) who developed a model that
could detect cyberbullying over multiple social
media platforms. To achieve this, we need to understand
if we can develop features for our model
that have strong predictive powers but are generalized
enough that they can still provide information
across different styles and domains of reviews
in various forums. This will be done by training
identical models on reviews from different forums
and attributing information gain to each feature to
identify the most promising features (Schouten et
al., 2016). The selected features will be used in
the final generalized model that will be then tested
on 3 different review forum datasets covering different
topics and domains evaluate how well these
general features perform.
## Relevance
A generalized model as we have proposed could
be a useful tool for many reasons. In today’s
world, review forums are a key tool for consumers
when making choices on products and services
they wish to purchase. For businesses as well,
these forums provide a fountain of information
on how their different products are received by
the population. This information can be beneficial
in product improvement as well as maintaining
and improving the brands image. However,
with multiple types of platforms (yelp, tripadvisor,
facebook, google, twitter, amazon, etc.) that consumers
can use, it can be difficult for businesses
to consolidate all the comments into one place to
glean the desired information. Additionally, while
some of these platforms are made explicitly to
have a rating system used alongside the review, not
all platforms have this in place. Without this rating
system such as is seen on yelp or amazon, it can
be difficult for a company to extract the sentiment
of whether or not a review is positive or negative.
Our model ideally would be able to help with this
issue by being able to scan through text from various
sources and format styles and provide an accurate
sentiment prediction of any text for a business.
This would help companies create a standardized
format in which they could understand the overall
sentiment of consumers.<br><br>
Another way this model can have impact is for
companies that are branching out into a new domain.
Since most sentiment analysis models are
domain specific, providing high accuracy classification
for a narrow scope. If a company who
specializes in selling electronics moves into selling
books, their prediction model for sentiment
on the reviews of electronic sales might be poorly
equipped to make predictions on book reviews.
This is where our generalized model would step in
to make predictions and tag the reviews as either
positive or negative. While incorrect tags might
happen more frequently with a generalized model,
it would allow for a company to be able to immediately
analyze the reviews in this new domain
before the company has enough data to create a
domain specific model. Correcting improperly labeled
tags will be much easier and less expensive
than manually labelling.
## Approach
### Data
To address the problem of creating a generalized
sentiment analysis model, we decided to take data
from two different review forums. To give us reviews
on a variety of different domains, we have
chosen the Yelp review and the IMDB datasets
from Kaggle. The latter dataset is a cleaned
dataset that consists of 50k different reviews of
movies that are tagged with the correct sentiment
for each review. This dataset is evenly divided
between positive and negative labels. The yelp
dataset is a larger dataset of over 5 million reviews
of 174,000 different businesses. The benefit
of this datasets is it allows us to get a variety
of reviews for different types of businesses to
help us generalize what features are best for correctly
tagging the reviews. Since this dataset is
too large for the computational power that we have
and significantly larger than the IMDB dataset,
we decided to sample randomly about 100,000 reviews.
To do this, we first removed the three-star
reviews from the overall dataset. Since yelp uses
a star rating system, 1 star being the poorest review
and 5 stars being the best, the 3 star reviews
can be seen as neutral reviews without strong positive
or negative opinion. Since the IMDB dataset
does not have neutral reviews, we have decided
to only work with 4 and 5 star reviews and 1 and
2 star reviews. Tagging them positive and negative
reviews respectively. Due to a large imbalance
between the positive and negative reviews in
the dataset, we separated the positive and negative
reviews into separate datasets. From each of these
datasets, we then randomly sampled 50,000 observations
and then combined those observations into
one yelp dataset.<br><br>
Due to the nature of this model only focusing on
English reviews, the two datasets needed to be
cleaned of any non-English reviews. The IMDB
dataset already only contained English reviews, so
this left only the yelp dataset to be cleaned. To accomplish
this, we used python’s langdetect library
which is ported from Google’s language-detection
code. This library helped identify the language
of review and then we kept only those labelled
as English. After the language cleaning, we have
our two cleaned datasets ready for feature engineering;
the IMDB dataset consisting of 50,000
reviews and the yelp dataset consisting of 99,579
reviews.
### Methods
####  Feature Extraction
Having found little literature on creating a generalized
sentiment analysis model, we decided to select features proposed by other papers that were
also building sentiment analysis models and using
information gain in the feature selection process
(Liu et al., 2014). These features were
then developed separately on the Yelp and IMDB
datasets.<br><br>
Since sentiment analysis has already been heavily
researched for many years, we are fortunate
to be able to stand on the shoulders of giants
when developing our own features. Our first set
of chosen features are borrowed from (Schouten
et al., 2016). When developing their model, they
found negation words being present or high counts
of positive or negative words, were important in
the sentiment prediction process (Liu et al., 2014)
observed that counting words per sentence, total
words per review, total punctuation and exclamations
per review were also features with high information
gain. Finally, as an addition of our own, we
have incorporated 2 and 3 character grams present
in the reviews and percentage of total words being
stop words for each review. Our decision to
use character grams rather than word grams is to
hopefully create more generalizability. Words can
carry specific meaning especially in certain contexts.
Therefore, using character grams instead of
word grams will hopefully remove the contextual
aspects that words but still keep the information
that the syntax of the text provides.<br><br>
To extract the exclamation count, punctuation
count, words per sentence, and words per review,
we simply ran different tokenizers over the reviews
to count the occurences of these different
aspects. To build our percent stop words feature,
we counted the frequency that stopwords
from the string library occur compared to the rest
of the words. For the negation present, positive
word count, and negative word count, we used
negation,positive, and negative word lists from the
General Inquirer Lexicon (Schouten et al., 2016).
The negation feature is binary indicating whether
or not any negation words are present in the text.
The positive and negative word features alongside
the 2 and 3 character gram features were passed
through a TFIDF vectorizer creating a feature for
each gram and word. The vectorization process
and the scaling process using StandardScaler for
the non vectorized features were fitted and transformed
on the training data and then only fitted on
the testing data. This provided an average 6,000
features the yelp and IMDB datasets each.<br>
With these two sets of features, we calculate the
information gain (IG) of each feature using the
mutual info classif from scikit learn (Kraskov et
al., 2004, Ross, 2014, Kozachenko & Leonenko,
1987). Using the two sets of features and their IGs
we filter out features that are nearly zero (less than
0.001) and features who don’t have similar IGs
across the two datasets. To do this, we took the absolute
difference between these identical features
in each dataset and kept features that had a difference
between zero and one standard deviation
above the mean. This left us with 1,486 features,
comprised of 912 character grams, 568 positive
and negative words, and the exclamation count,
punctuation count, words per sentence, words per
review, negation, and percentage of stop words
features.
#### Machine Learning
In our research of sentiment analysis models, we
found that Support Vector Machines and Naive
Bayes classifiers continuously stood out as top
performers for labelling text data with sentiments
(Liu et al., 2014, Schouten et al., 2016, Tripathy
et al., 2014). Of these two classifiers, we decided
to choose a SVM algorithm with stochastic
gradient descent as it would perform best of the
two with our vectorized features. Since our research
is focused primarily on the effect of our features
rather than the parameters of the model, we
passed our sparse matrix through the out-of-thebox
SGDClassifier from scikit-learn without any
changed parameters.
## Evaluation
To assess our model’s performance, we decided
to use the accuracy of the algorithm’s labeling
abilities. This evaluation method was chosen due
to our balanced datasets and both classes being
equally important to predict correctly. Once our
features had been created and pared down through
our feature engineering process, we then used the
selected features in our pipeline on the Yelp and
IMDB datasets. Additionally as a final test, we ran
the model on an unseen dataset of 8,000 reviews
from Coursera that were cleaned and balanced in
a similar method to the Yelp dataset.
## Results
Results obtained from our model were overall positive.
On the Yelp, IMDB, and Coursera datasets,we achieved accuracies of 91.6%, 84%, and 89.7%
respectively. These are promising results showing
that with generalized features, models can still obtain
strong results. While there is a large variance
in the results among the different datasets, this
type of performance if consistent across untested
platforms would provide a strong base for companies
looking to achieve any of the goals mentioned
in our section 3. When extracting the IG of each
feature, the features that had the highest IG were
different 2 and 3 character grams. The positive and
negative words along with the other features explained
above had relevant IGs but were substantially
smaller than those of the character grams.
This can show that specific character grams can be
as important as the semantics of words in a sentence
as well.
## Future Directions
While our model achieved strong results, there are
improvements that could be made to make it a
more robust model before being applied in a real
business setting. The first improvement that could
be made is to modify the model to include a neutral
rating class. Many review forums our assessed
out of 5 for rating, leaving 3 stars for example as
a middle ground. We did not include this in our
model because one of our IMDB dataset was only
binary, but neutral ratings hold a lot of information
for business and researchers and would be the
next step in enhancing this model. The other improvement
that could be made is to train and improve
this model on imbalanced datasets. We used
datasets balanced between positive and negative
reviews to mimic the IMDB dataset but commonly
in review forums, there is a far higher amount of
positive reviews than negative. Creating a model
that would perform well on unbalanced or balance
data using positive, negative, and neutral classes
could improve this model significantly and potentially
make it ready to be used in a live setting.
In regards to fine-tuning the model, further exploration
of the features might provide insight allowing
for a more condensed feature set. Our feature
set is pretty small but results from (Schouten et
al., 2016) suggest that even just using the top 1%
of features with the highest information gain can
result in strong results. Further paring down of
features could make the model more agile, allowing
it to be deployed quickly and still providing
high accuracy.
## References
Schouten, K., Frasincar, F., & Dekker, R. 2016,
June An information gain-driven feature study
for aspect-based sentiment analysis, International
Conference on Applications of Natural
Language to Information Systems (pp. 48-59).
Springer, Cham.<br><br>
Agrawal, S., & Awekar, A. 2018, March Deep learning
for detecting cyberbullying across multiple
social media platforms, European Conference
on Information Retrieval (pp. 141-153). Springer,
Cham.<br><br>
Nababan, A. A., & Sitompul, O. S. 2018, April Attribute
Weighting Based K-Nearest Neighbor Using
Gain Ratio., Journal of Physics: Conference
Series (Vol. 1007, No. 1, p. 012007). IOP Publishing.<br><br>
Taneja, S., Gupta, C., Goyal, K., & Gureja, D. 2014,
February An enhanced k-nearest neighbor algorithm
using information gain and clustering,
2014 Fourth International Conference on Advanced
Computing & Communication Technologies
(pp. 325-329). IEEE.<br><br>
Shah, D., Isah, H., & Zulkernine, F. 2018, December
Predicting the effects of news sentiments on the
stock market., In 2018 IEEE International Conference
on Big Data (Big Data) (pp. 4705-4708).
IEEE.<br><br>
Liu, C., Guo, C., Dakota, D., Rajagopalan, S., Li, W.,
K¨ubler, S., & Yu, N. 2014 “My Curiosity was
Satisfied, but not in a Good Way”: Predicting
User Ratings for Online Recipes., (2014).<br><br>
Tripathy, A., Agrawal, A., & Rath, S. K. 2015 Classication
of Sentimental Reviews Using Machine
Learning Techniques., Procedia Computer Science,
57, 821-829.<br><br>
Wilson, T., Wiebe, J., & Hoffmann, P. 2009. Recognizing
contextual polarity: An exploration of features
for phrase-level sentiment analysis., Computational
linguistics, 35(3), 399-433.<br><br>
Feldman, R. 2013 Techniques and applications for
sentiment analysis., Communications of the
ACM, 56(4), 82-89.<br><br>
A. Kraskov, H. Stogbauer and P. Grassberger 2004
“Estimating mutual information”, Phys. Rev. E
69, 2004.<br><br>
B. C. Ross 2014 “Mutual Information between Discrete
and Continuous Data Sets”. , PLoS ONE
9(2), 2014<br><br>
L. F. Kozachenko, N. N. Leonenko 1987 “Sample Estimate
of the Entropy of a Random Vector:, Probl.
, Peredachi Inf., 23:2 (1987), 9-16
