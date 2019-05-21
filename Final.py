
# Group Members
# =============================================================================
# 1. Dhruv Mehta
# 2. Nitanshu Shrivastava
# 3. Pikaso Dutta
# 4. Shreyas Wattamwar
# =============================================================================

# Set Directory
import os
os.chdir('D:/Stevens/595/Mid-Term') 


# In[1]: Login to facebook

# Update Facebook Id & Pass in User_Id.txt
from Functions import login
login()


# In[2]: Scrap Data

# Load UDF to Scrap: O/p .html scrap file
from Functions import scrapData
scrapData("https://www.facebook.com/pg/ATT/community/?ref=page_internal",'Att_reviews.html')
scrapData("https://www.facebook.com/pg/TMobile/community/?ref=page_internal",'T-Mobile_reviews.html')


# In[3]: Fetch Review

# Load UDF to fetch reviews: O/p .txt file
from Functions import fetchReview
reviews_att = fetchReview("Att_reviews.html",'review_data_att.txt')
reviews_Tmobile = fetchReview("T-Mobile_reviews.html",'review_data_T-Mobile.txt')


# In[4]: Analyze review

# Compare with pre-defined +/- words
from Functions import directData
reviews_prediction_att = directData(reviews_att)
reviews_prediction_Tmobile = directData(reviews_Tmobile)


# In[5]: Vader Analysis

# Get +/- rating for each review
from Functions import sentiment_analysis_VADERAnalyser
summary_att,sentences_att,label_att=sentiment_analysis_VADERAnalyser(reviews_att)
summary_Tmobile,sentences_Tmobile,label_Tmobile=sentiment_analysis_VADERAnalyser(reviews_Tmobile)


# In[6]: Plot Analysis

# Bar graph for +/-/0 reviews
from Functions import plot_Sentiment
plot_Sentiment(summary_att, 'AT&T')
plot_Sentiment(summary_Tmobile, 'T-Mobile')


# In[7]: Compare Sentiments

# Compare AT&T with T-Mobile
from Functions import compare_sentiments
compare_sentiments(summary_att, summary_Tmobile)


# In[8]: Clean Sentences


from Functions import cleanup_data
clean_sentences_att = cleanup_data(sentences_att)
clean_sentences_Tmobile = cleanup_data(sentences_Tmobile)


# In[9]: NB Classifier


from Functions import multinomialNB
Accuracy_att = multinomialNB(clean_sentences_att,label_att)
Accuracy_Tmob = multinomialNB(clean_sentences_Tmobile,label_Tmobile)


