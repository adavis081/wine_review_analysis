"""
The purpose of this script is to investigate the potential relationship between price and rating. 
Also, to explore how the descriptions relate to the ratings.
"""

### Import all needed packages

# for data manipulation 
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

# for stats
import scipy.stats as stats

# for nlp and text
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud

# for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report




# Import data
reviews = pd.read_csv("Data/reviews_clean.csv")

# Distribution of ratings
print(reviews['points'].describe())


# histogram of points
figsize(7, 5)
plt.hist(reviews['points'], color='blue', edgecolor='black', bins= 20)
plt.xlabel('Points')
plt.ylabel('Point Frequency')
plt.title('Points Frequencies')

plt.savefig("Visualizations/hist_points_original.png")


"""
Points have a normal distribution with a mean of 88 and a standard deviation of 3.
Min = 80 and Max = 100. Next, we will investigate if price has an effect on points.
"""

# checking the correlation coefficient between price and points
print(np.corrcoef(reviews['price'], reviews['points'])) # 0.40166


"""
The correlation between price and points is weak. We cannot say if expensive wines are scored 
differently than cheap wines. Next, we will split the data between wines that were priced 1-60
and 61-200. We can conduct an ANOVA test to determine if the average rating between these
two groups is different.
"""

# Null Hypothesis: The average points given between the two groups are the same
# Alt Hypothesis: The average points given between the two groups are different


# split the data in the two price groups
cheap = reviews[reviews['price'] < 61]
expensive = reviews[reviews['price'] >= 61]

# conduct ANOVA test
fvalue, pvalue = stats.f_oneway(cheap['points'], expensive['points'])
print(fvalue, pvalue)


"""
With a p-value of 0, we cannot reject our null hypothesis that the average rating between the
two groups are the same. Therefore, we can say that price has no effect on points. 
"""

###########################################################################################


"""
Next, we can use the 'description' field to learn more betwen wines scored lowly and highly.
First, we must clean the text of the descriptions.
"""

# lower case all description text
reviews['description_clean'] = reviews['description'].str.lower()


# create function for punctuation removal:
def remove_punctuations(text):
    for char in string.punctuation:
        text = text.replace(char, '')
    return text


# apply the function
reviews['description_clean'] = reviews['description_clean'].apply(remove_punctuations)


# remove stop words
stop = stopwords.words('english')
more_words = ['much', 'more', 'very', 'include', 'many', 'heres', 'theres', 'isnt', 'alongside', 'offering', 'overly']
all_stop_words = stop + more_words

reviews['description_clean'] = reviews['description_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in all_stop_words]))



"""
Now that the text is cleaned up, let's first generate some word clouds for different ratings.
"""


# split data by points (80-90, 91-100) and examine the most common words used in the descriptions
scored_80_90 = reviews[reviews['points'] < 91]
scored_91_100 = reviews[reviews['points'] >= 91]


# generate wordcloud for 80-90
wordcloud_80_90 = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(scored_80_90['description_clean'].values[0])

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_80_90)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.savefig("Visualizations/wordcloud_80_90.png")


# generate wordcloud for 91-100
wordcloud_90_100 = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(scored_91_100['description_clean'].values[0])

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_90_100)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.savefig("Visualizations/wordcloud_90_100.png")




"""
The wordclouds between wines scored 'highly' and 'lowly' don't seem to tell us anything 
worthwhile. The most common words used are not objectivley positive or negative. For lowly
scored wines, the most common words are not very expressive and just list some of the characteristics.
For highly scored wines, we see words like 'vibrant' and 'crisp'. All this tells us is that 
wines that judges enjoyed more were scored higher. Not to surprising, but interesting to see.

For one more shot, let's see if we can use the descriptions to accuratley predict points!
"""

# bin the reviews by quartiles 
reviews['label'] = pd.qcut(reviews['points'],
                           q=[0, 0.25, 0.5, 0.75, 1],
                           labels=['Fair', 'Good', 'Great', 'Excellent'])

print(reviews['label'].value_counts())



# train test split (70% train - 30% test)
X_train, X_test, y_train, y_test = train_test_split(reviews['description_clean'], reviews['label'], test_size=0.3, random_state=123)



# initilize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_vectors = tfidf_vectorizer.transform(X_test)



# initilize random forest classifier 
classifier = RandomForestClassifier()


# fit model to data and make preditions on the test set
classifier.fit(tfidf_train_vectors,y_train)
y_pred = classifier.predict(tfidf_test_vectors)


# print results
print(classification_report(y_test,y_pred))


"""
Results are poor. Average accuracy of 60%. Precision and Recall scores never exceed 0.81. Such 
a poor accuracy confirms the suspicion from the wordclouds, that the descriptions don't correlate
to the scores. They are likely just describing flavor, and overall are not positive or negative,
just descriptive.
"""


"""
Conclusion: It is unlikely that there is a correlation between scores and price, as well as
score and description. Scores are likely highly subjective, and depend on factors outside this
data set.
"""
