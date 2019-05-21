# Functions

# 1. Login to webpage
# 2. Scrap data from webpage
# 3. Get reviews from scrapdata
# 4. Sentimental analysis setup
# 5. Run sentimental analysis
# 6. Vader Sentimental analysis setup
# 7. Plot Analysis
# 8. Compare Plots
# 9. Clean Sentences
# 10. Naive Based Classifier 




# Import required libraries

import re
import numpy as np
import time, codecs
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

driver = webdriver.Chrome()


# 1. Login to webpage
def login():
        
    # Import username and password from text file
    Input = open("User_Id.txt","r") 
    data = Input.readlines()
    username = data[1][5:-1].lstrip()
    password = data[2][5:-1].lstrip()

    #access website
    try:
        driver.get('https://www.facebook.com/')
        
        #Accessing Login frame
        form=driver.find_element_by_id('login_form')
        form.click()
        
        #Entering email details
        email = form.find_element_by_id('email')
        email.send_keys(username)
        #time.sleep(1)
        
        #Entering password details
        pwd = form.find_element_by_id('pass')
        pwd.send_keys(password)
        time.sleep(1)
        
        #Clicking the login button
        button=WebDriverWait(driver, 1000).until(EC.element_to_be_clickable((By.ID, 'loginbutton')))
        button.click()
        
    except Exception as e:
        print('Exception encountered during Login')
        print(e)
   

     
# 2. Scrap data from webpage
def scrapData(url, html):
    
    # url:  Url to scrap
    # html: Scraped file saved to dir in .html 
    
    try:
        driver.get(url)
        time.sleep(3)

        # Selenium script to scroll to the bottom, wait 3 seconds for the next batch of data to load, then continue scrolling.  It will continue to do this until the page stops loading new data.
        lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        noOfPageScrolls=0
        while(noOfPageScrolls < 100):
            
            time.sleep(3)
            lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            print(html,'-',lenOfPage,'-',noOfPageScrolls)
            noOfPageScrolls=noOfPageScrolls+1

        # Now that the page is fully scrolled, grab the source code.
        source_data = driver.page_source

        # Passing page source into BeautifulSoup to start parsing
        bs_data = bs(source_data, features="lxml")

    except Exception as e:
        print('Exception getting the Page Source')
        print(e)
    try:
        with codecs.open(html,'w',encoding='utf8') as fw: fw.write(str(bs_data))
        fw.close()
    except Exception as e:
        print('Exception writing the Page Source into File')
        print(e)
    
    return bs_data

    

# 3. Get reviews from scrapdata(in HTML)
def fetchReview(scrapfile, text_output):
    
    # scrapfile: O/p of UDF scrapdata in .html
    # text_output: Get reviews from scrapdata and export .txt
    
    reviews =[]
    with open(scrapfile, 'rb') as html:
        soup = bs(html, "lxml")
    #Beautiful soup to fetch the review
    reviewChunk=soup.findAll('div',{'class':(re.compile("_5pbx userContent"))}) 
    #print(reviewChunk)    
    with codecs.open(text_output, 'w',encoding='utf8') as fw: 
        if reviewChunk: 
            for review in reviewChunk:
                r=review.text.strip()
                fw.write(r)
                fw.write('\r\n')
                reviews.append(r)
    fw.close()
    return(reviews)
    

# 4. Sentimental analysis setup   
def sentiment_analysis(text, positive_words, negative_words):
             
    sentiment=None
    
    posWordCount = 0
    negWordCount = 0

    tokens =  nltk.word_tokenize(text)

    print("\n")
    
    for idx, token in enumerate(tokens):
        if token in positive_words:
            if idx>0:
     #    - Positive words:
     #  * a positive word not preceded by a negation word (i.e. not, n't, no, cannot, neither, nor, too)
                if tokens[idx-1] not in negative_words:
                    posWordCount += 1
                else:
      #- Negative words:
      # * a positive word preceded by a negation word
                    negWordCount += 1
            else:
                posWordCount += 1
        elif (token in negative_words):
            if idx>0:
       #- Negative words:
      # * a negative word not preceded by a negation word
                if tokens[idx-1] not in negative_words:
                    negWordCount += 1
                else:
    # - Positive words:
      # * a negative word preceded by a negation word (ex -not bad)
                    posWordCount += 1
            else:
                negWordCount += 1
    
    if(posWordCount > negWordCount):
        sentiment = "positive"
    elif(posWordCount <= negWordCount):
        sentiment = "negative"
    
    return sentiment


# 5. Run sentimental analysis
def directData(reviews):
    from Functions import sentiment_analysis
    #with open("review_data.txt") as f:
    sentiments_prediction=[]
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
    for review in reviews:
        sentiments_prediction.append(sentiment_analysis(review,positive_words,negative_words))
    return(sentiments_prediction)
    
    
    
# 6. Vader Sentimental analysis setup  
def sentiment_analysis_VADERAnalyser(sentences):
    import nltk
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    summary = {"positive":0,"neutral":0,"negative":0}
    label=[]
    "positive :1, negative :2,neutral :3"
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        if(ss["compound"]>=0.02):
            summary["positive"] +=1
            label.append(1)
        elif( ss["compound"]<= -0.02):
            summary["negative"] +=1
            label.append(2)
        else:
            summary["neutral"] +=1
            label.append(3) 
    return(summary,sentences,label)


# 7. Plot Analysis
def plot_Sentiment(summary, title):
    print("h:")
    my_colors = ['b','yellow','r']
    positive = summary["positive"]
    negative = summary["negative"]
    neutral = summary["neutral"]
    objects=("positive","neutral","negative")
    y_pos = np.arange(len(objects))
    performance=[positive, neutral, negative]
    plt.figure(figsize=(6,6))
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.bar(y_pos, performance, align='center', color=my_colors)
    plt.title(title, fontsize=20)
    plt.xticks(y_pos, objects)
    plt.show()


# 8. Compare Plots  
def compare_sentiments(summary_att, summary_Tmobile):
    sentiments = 3
    att = (summary_att['positive'],summary_att['negative'], summary_att['neutral'])
    Tmobile = (summary_Tmobile['positive'],summary_Tmobile['negative'], summary_Tmobile['neutral'])
    fig, ax = plt.subplots()
    index = np.arange(sentiments)
    bar_width = 0.35
    opacity = 0.8
    
    att_plot = plt.bar(index, att, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='At&t')

    Tmobile_plot = plt.bar(index + bar_width, Tmobile, bar_width,
                     alpha=opacity,
                     color='yellow',
                     label='Tmobile')
    
    plt.xlabel('Sentiments')
    plt.ylabel('Scores')
    plt.title('Sentiments of each company')
    plt.xticks(index + bar_width, ('positive', 'negative','neutral'))
    plt.legend()
    plt.show()    
    

# 9. Clean Sentences
def cleanup_data(sentences):
    clean_sentences = []
    for sentence in sentences:
        sentence=re.sub(r"http.?://[^\s]+[\s]?","",sentence)
        sentence=re.sub(r"@[^\s]+[\s]?","",sentence)
        sentence=re.sub(r"\s?[0-9]+\.?[0-9]*","",sentence)
        sentence=re.sub(r"[^a-zA-Z0-9 ]","",sentence)
        clean_sentences.append(sentence)

    return clean_sentences

    
# 10. Naive Based Classifier   
def multinomialNB(clean_sentences,label):

    # initialize the TfidfVectorizer 
    tfidf_vect = TfidfVectorizer()
    # with stop words removed
    tfidf_vect = TfidfVectorizer(stop_words = "english") 
    # generate tfidf matrix
    dtm = tfidf_vect.fit_transform(clean_sentences).toarray()
    X_train, X_test, y_train, y_test = train_test_split(dtm, label,test_size = 0.20, random_state = 0)
    clf = MultinomialNB().fit(X_train,y_train)
    accuracy = [cross_val_score(clf, dtm, label, cv=80)]
    
    return (str(round(np.array(accuracy[0]).mean()*100,2))+'%') 

