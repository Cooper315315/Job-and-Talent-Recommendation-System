
#%%
# import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import os
import base64
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
import pandas as pd 
from nltk.collocations import *
from nltk.corpus import words
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
#%%
# Ignore warning
# st.set_option('deprecation.showPyplotGlobalUse', False)
# # set wide layout
# st.set_page_config(layout="wide")
#%%
# (OCR function)
# def extract_data(feed):
#     text=''
#     with pdfplumber.open(feed) as pdf:
#         pages = pdf.pages
#         for page in pages:
#             text+=page.extract_text(x_tolerance=2)
#     return text
#%%
def app():
    # Title & select boxes--------------------------display##
    st.title('Candidate Recommendation')
    # cv=st.file_uploader('Upload your CV', type='pdf')
    c1, c2 = st.beta_columns((3,2))
    # upload jd + turn pdf to text------------------display##
    #try:
    # jd=c1.file_uploader('Upload your JD', type='pdf')
    # number of cv recommend slider------------------display##
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=3, max_value=10, step=1)
    #%%
    # text area for enter JD
    # default_value_goes_here='hi'
    jd = c1.text_area("paste your job post here")
    #%%
    if jd is not None:
        # jd_text=extract_data(jd)
    #%%
        # (NLP funtion)
        # import stop word lists for NLP function
        # Locations
        @st.cache #method to get data once and store in cache.
        def get_location():
            f1=open('./hk_districts.txt','r', errors = 'ignore')
            text1=f1.read()
            return word_tokenize(text1.replace("\n", " "))
        locations = get_location()
        # (Additional stopwords function)
        @st.cache #method to get data once and store in cache.
        def get_stopwords():
            f2=open('./stopwords.txt','r', errors = 'ignore')
            text2=f2.read()
            return word_tokenize(text2.replace("\n", " "))
        stopwords_additional = get_stopwords()
        #%%
        #(NLP keywords function)
        @st.cache
        def nlp(x):
            word_sent = word_tokenize(x.lower().replace("\n",""))
            _stopwords = set(stopwords.words('english') + list(punctuation)+list("●")+list('–')+list('’')+locations+stopwords_additional)
            word_sent=[word for word in word_sent if word not in _stopwords]
            lemmatizer = WordNetLemmatizer()
            NLP_Processed_JD = [lemmatizer.lemmatize(word) for word in word_tokenize(" ".join(word_sent))]
        #     return " ".join(NLP_Processed_CV)
            return NLP_Processed_JD
        #%%
        @st.cache
        def remove_stuff(jd):
            jd_clean = jd.replace("\xa0", "").replace("/", "").replace(".", ". ").replace("●", "")
            return jd_clean

        #%%
        # (NLP keywords for JD workings)
        NLP_Processed_JD=nlp(jd)
        # st.text(NLP_Processed_JD)
        #create jd df
        jd_df=pd.DataFrame()
        # jd_df['hi']=['I']
        jd_df['jd']=[' '.join(NLP_Processed_JD)]
        # st.dataframe(jd_df)

        @st.cache
        def get_recommendation(top, df, scores):
            recommendation = pd.DataFrame(columns = ['cv_id','phone number','email','pdf','score'])
            count = 0
            for i in top:
                recommendation.at[count, 'cv_id'] = df.index[i]
                recommendation.at[count, 'phone number'] = df['phone number'][i]
                recommendation.at[count, 'email'] = df['email'][i]
                recommendation.at[count, 'pdf'] = df['pdf'][i]
                recommendation.at[count, 'score'] =  scores[count]
                count += 1
            return recommendation
        #%%
        @st.cache 
        def get_cv(): #cleaned, processed, nlped cv content
            url='./Data/cv_nlp_pdf.csv'
            return pd.read_csv(url) 
        # db_expander = st.beta_expander(label='Submitted resume:')
        # with db_expander:
        df = get_cv()
        #     st.dataframe(df[['phone number','email']])
    #%%
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        @st.cache 
        def TFIDF(cv,jd):
            tfidf_vectorizer = TfidfVectorizer()
            # TF-IDF Scraped data
            tfidf_jobid = tfidf_vectorizer.fit_transform(cv)

            # TF-IDF CV
            user_tfidf = tfidf_vectorizer.transform(jd)

            # Using cosine_similarity on (Scraped data) & (CV)
            cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf,x),tfidf_jobid)

            output2 = list(cos_similarity_tfidf)
            return output2
            
        tf = TFIDF(df['cv'],jd_df['jd'])

        # show top cv recommendations using TF-IDF
        top = sorted(range(len(tf)), key=lambda i: tf[i], reverse=True)
        list_scores = [tf[i][0][0] for i in top]
        tf_df=get_recommendation(top,df, list_scores)
    #%%
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        @st.cache 
        def count_vectorize(cv,jd):
            # CountV the scraped data
            count_vectorizer = CountVectorizer()
            count_jobid = count_vectorizer.fit_transform(cv) #fitting and transforming the vector

            # CountV the cv
            user_count = count_vectorizer.transform(jd)
            cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
            output3 = list(cos_similarity_countv)
            return output3
        countv = count_vectorize(df['cv'],jd_df['jd'])

        top = sorted(range(len(countv)), key=lambda i: countv[i], reverse=True)
        list_scores = [countv[i][0][0] for i in top]
        cv_df=get_recommendation(top, df, list_scores)
    #%%
        from sklearn.neighbors import NearestNeighbors
        @st.cache 
        def KNN(cv, jd):
            tfidf_vectorizer = TfidfVectorizer()

            n_neighbors = 40
            KNN = NearestNeighbors(n_neighbors, p=2)
            KNN.fit(tfidf_vectorizer.fit_transform(cv))
        #     NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv), return_distance=True)
            NNs = KNN.kneighbors(tfidf_vectorizer.transform(jd))
            top = NNs[1][0][1:]
            index_score = NNs[0][0][1:]

            knn = get_recommendation(top, df, index_score)
            return knn

        knn_df = KNN(df['cv'],jd_df['jd'])
        # knn = KNN(df_Accountant['job description'], df3['Resume'])
        # knn.sort_values(by = 'score', ascending = True)
    #%%
        merge1 = knn_df[['cv_id','phone number','email', 'pdf','score']].merge(tf_df[['cv_id','score']], on= "cv_id")
        final = merge1.merge(cv_df[['cv_id','score']], on = "cv_id")
        final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF","score": "CV"})
    #%%
        # Scale it
        from sklearn.preprocessing import MinMaxScaler
        slr = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

        # Multiply by weights
        final['KNN'] = (1-final['KNN'])/3
        final['TF-IDF'] = final['TF-IDF']/3
        final['CV'] = final['CV']/3
        final['Final'] = final['KNN']+final['TF-IDF']+final['CV']
        final.sort_values(by="Final", ascending=False, inplace=True)
        final.reset_index(inplace=True)
        # st.dataframe(final)
    # %%
        final1=final[['cv_id','phone number','email','pdf','Final']].head(no_of_cv)
    #%%
        db_expander = st.beta_expander(label='CV recommendations:') 
        with db_expander:
            # st.dataframe(final[['cv_id','phone number','email']].head(no_of_cv))
        
            no_of_cols=3
            cols=st.beta_columns(no_of_cols)
            for i in range(0, no_of_cv):
                cols[i%no_of_cols].text(f"CV ID: {final1['cv_id'][i]}")
                cols[i%no_of_cols].text(f"Phone no.: {final1['phone number'][i]}")
                cols[i%no_of_cols].text(f"Email: {final1['email'][i]}")
                cols[i%no_of_cols].text('Candidate CV:')
                with open(final1['pdf'][i],"rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">' 
                # pdf_display = '<embed src="https://drive.google.com/file/d/1tNAi4a4vEotPn7FBu_HJZKzICSKyWlAU/view?usp=sharing" width="700" height="1000" type="application/pdf">' 
                cvID=final1['cv_id'][i]
                show_pdf=cols[i%no_of_cols].button(f"{cvID}.pdf")
                if show_pdf:
                    st.markdown(pdf_display, unsafe_allow_html=True)
                cols[i%no_of_cols].text('___________________________________________________')