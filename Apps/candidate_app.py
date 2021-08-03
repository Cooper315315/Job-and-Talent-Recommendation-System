# [theme]
# primaryColor="#87cefa"
# backgroundColor="#f0f8ff"
# secondaryBackgroundColor="#a0acf1"
# textColor="#101010"

#%%
# !pip install pdfplumber 
# !pip install ocrmypdf
# !pip install plotly
#%%
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
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
import re
import plotly.express as px
import time
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#%%
# Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)
# set wide layout
st.set_page_config(layout="wide")
#%%
def app():
    # (OCR function)
    def extract_data(feed):
        text=''
        with pdfplumber.open(feed) as pdf:
            pages = pdf.pages
            for page in pages:
                text+=page.extract_text(x_tolerance=2)
        return text
    #%%
    # Title & select boxes--------------------------display##
    st.title('Job Recommendation')
    # cv=st.file_uploader('Upload your CV', type='pdf')
    c1, c2 = st.beta_columns((3,2))
    # upload cv + turn pdf to text------------------display##
    #try:
    cv=c1.file_uploader('Upload your CV', type='pdf')
    #except ValueError:
    #    st.error('Please enter a valid input')
    #%%
    # career level-----------------------------------display##
    #version 1--

    levels = ["Entry Level","Middle", "Senior", "Top", "Not Specified"]
    CL = c2.multiselect('Career level', levels, levels)
        
        


    #version 2--
    # CL = c2.selectbox("Career level", ("Entry Level","Middle", "Senior", "Top"))
    #%%
    # number of job recommend slider------------------display##
    no_of_jobs = st.slider('Number of Job Recommendations:', min_value=20, max_value=100, step=10)
    #def Job_recomm(x):
    #    final_ = final[['title','career level','company','location','industry']]
    #    cl_select = final_[final_["career level"]==x]
    #    return cl_select
    #result_jd = Job_recomm(CL)

    if cv is not None:
        cv_text = extract_data(cv)
            # print(cv_text)


    #        def Job_recomm(x):
    #            final_ = final[['title','career level','company','location','industry']]
    #            cl_select = final_[final_["career level"]==x]
    #            return cl_select
    #
    #        result_jd = Job_recomm(CL)




        #----------------------------workings---------------------#

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
            NLP_Processed_CV = [lemmatizer.lemmatize(word) for word in word_tokenize(" ".join(word_sent))]
        #     return " ".join(NLP_Processed_CV)
            return NLP_Processed_CV
        #%%
        # (NLP keywords for CV workings)
        try:
            NLP_Processed_CV=nlp(cv_text)
        except NameError:
            st.error('Please enter a valid input')
        #NLP_Processed_CV=func(cv_text)
        #%%
        # put CV's keywords into dataframe
        df2 = pd.DataFrame()
        # append columns to an empty DataFrame
        df2['title'] = ["I"]
        df2['job highlights'] = ["I"]
        df2['job description'] = ["I"]
        df2['company overview'] = ["I"]
        df2['industry'] = ["I"]
        # Compare with the key words from CV only
        # df2['All'] = " ".join(Key_word_from_CV)

        # Compare with the entire CV
        df2['All'] = " ".join(NLP_Processed_CV)
        #%%
        # import whole nlp csv
        @st.cache #method to get data once and store in cache.
        def get_jobcsv():
            url='./Data/whole-v7-nlp.csv'
            return pd.read_csv(url)
        df= get_jobcsv()

        #%%
        # recommendation function
        @st.cache
        def get_recommendation(top, df, scores):
            recommendation = pd.DataFrame(columns = ['JobID',  'title', 'career level', 'company', 'industry', 'salary', 'location', 'webpage','score'])
            count = 0
            for i in top:
        #         recommendation.at[count, 'ApplicantID'] = u
                recommendation.at[count, 'JobID'] = df.index[i]
                recommendation.at[count, 'title'] = df['title'][i]
                recommendation.at[count, 'career level'] = df['career level'][i]
                recommendation.at[count, 'company'] = df['company'][i]
                recommendation.at[count, 'industry'] = df['industry'][i]
                recommendation.at[count, 'salary'] = df['salary'][i]
                recommendation.at[count, 'location'] = df['location'][i]
                recommendation.at[count, 'webpage'] = df['webpage'][i]
                recommendation.at[count, 'score'] =  scores[count]
                count += 1
            return recommendation

        
        #%%
        # TF-IDF function
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        @st.cache
        def TFIDF(scraped_data, cv):
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            # TF-IDF Scraped data
            tfidf_jobid = tfidf_vectorizer.fit_transform(scraped_data)
            # TF-IDF CV
            user_tfidf = tfidf_vectorizer.transform(cv)
            # Using cosine_similarity on (Scraped data) & (CV)
            cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf,x),tfidf_jobid)
            output2 = list(cos_similarity_tfidf)
            return output2  # what does it return?
        output2 = TFIDF(df['All'], df2['All'])
        #%%
        # show top job recommendations using TF-IDF
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:1000]
        list_scores = [output2[i][0][0] for i in top]
        TF=get_recommendation(top,df, list_scores)

        # st.dataframe(TF) #####Show TF
        #%%
        # Count Vectorizer function

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        @st.cache
        def count_vectorize(scraped_data, cv):
            # CountV the scraped data
            count_vectorizer = CountVectorizer()
            count_jobid = count_vectorizer.fit_transform(scraped_data) #fitting and transforming the vector
            # CountV the cv
            user_count = count_vectorizer.transform(cv)
            cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
            output3 = list(cos_similarity_countv)
            return output3
        output3 = count_vectorize(df['All'], df2['All'])
        #%%
        # show top job recommendations using Count Vectorizer
        top = sorted(range(len(output3)), key=lambda i: output3[i], reverse=True)[:1000]
        list_scores = [output3[i][0][0] for i in top]
        cv=get_recommendation(top, df, list_scores)

        # st.dataframe(cv) ######### SHOW CV
        #%%
        # KNN function
        from sklearn.neighbors import NearestNeighbors

        @st.cache   
        def KNN(scraped_data, cv):
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            n_neighbors = 1000
            KNN = NearestNeighbors(n_neighbors, p=2)
            KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
            NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv), return_distance=True)
            top = NNs[1][0][1:]
            index_score = NNs[0][0][1:]
            knn = get_recommendation(top, df, index_score)
            return knn
        knn = KNN(df['All'], df2['All'])

        # st.dataframe(knn) ############ SHOW KNN
        #%%
        # Combine 3 methods into a dataframe
        merge1 = knn[['JobID','title','career level','company','location','industry','salary','webpage','score']].merge(TF[['JobID','score']], on= "JobID")
        final = merge1.merge(cv[['JobID','score']], on = "JobID")
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
        final.sort_values(by="Final", ascending=False, inplace=True) #make silde bar to change top N recommendations
        
        #job recommendations after career level & no. of jobs filter
        # @st.cache
        # def Job_recomm(x):
        #     final_ = final[['JobID','title','career level','company','location','industry']]
        #     st.dataframe(final_)
        #     # cl_select = final_[final_["career level"]==CL]
        #     st.dataframe(cl_select)
        #     return cl_select

        def Job_recomm(x):
            final_ = final[['JobID','title','career level','company','location','industry','salary', 'webpage']]
            selected_levels = final_['career level'].isin(CL)
            cl_select = final_[selected_levels]
            return cl_select
        
        # result_jd = Job_recomm(CL)
        result_jd = final
        final_jobrecomm =result_jd.head(no_of_jobs)

        #% 





        
        #--------------------workings end-------------------------#

        # Map code ---------------------------------------display#
        #import geopandas as gpd
        import geopy
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        import matplotlib.pyplot as plt
        import folium
        from folium.plugins import FastMarkerCluster
        from streamlit_folium import folium_static
        import folium
        from geopy.extra.rate_limiter import RateLimiter
        import numpy as np
        from streamlit_folium import folium_static

        #%%

        # Creating Map Plot

        #Make copy of recommendation DF
        df3 = final_jobrecomm.copy()
        # st.dataframe(df3)
        df3.fillna("Hong Kong", inplace=True)

        # Adding HK to location
        df3['location'] = df3['location'] + ',' + " Hong Kong"

        # Creating new DF with location and Job count for area
        rec_loc = df3.location.value_counts()
        locations_df = pd.DataFrame(rec_loc)
        locations_df.reset_index(inplace=True)

        #Removing word "Area" as impact Geopy
        locations_df['index'] = locations_df['index'].apply(lambda x: x.replace("Area", "") if "Area" in x else x)

        #Adding request limit as 1 to follow guidelines
        locator = Nominatim(user_agent="myGeocoder")
        geocode = RateLimiter(locator.geocode, min_delay_seconds=1) #1 second per api request

        #Extracting lat, long, alt
        locations_df['loc_geo'] = locations_df['index'].apply(geocode)
        locations_df['point'] = locations_df['loc_geo'].apply(lambda loc: tuple(loc.point) if loc else None)

        # split point column into latitude, longitude and altitude columns
        locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(locations_df['point'].tolist(), index=locations_df.index)

        #dropping any null values from lat / long

        locations_df.dropna(subset=['latitude'], inplace=True)
        locations_df.dropna(subset=['longitude'], inplace=True)

        #Set start location for map
        folium_map = folium.Map(location=[22.3186,114.1850],
                                zoom_start=11,
                                tiles= "cartodbdark_matter")


        #Adding points to map
        for lat, lon, ind, job_no in zip(locations_df['latitude'], locations_df['longitude'], locations_df['index'], locations_df['location']):
            label = folium.Popup("Area: " + ind + "<br> Number of Jobs: " + str(job_no), max_width=500)
            folium.CircleMarker(
                [lat, lon],
                radius=10,
                popup=label,
                fill = True,
                color='cadetblue',
                fill_col = "lightblue",
                icon_size = (150,150),
                ).add_to(folium_map)

        #%%
        # qualification bar chart
        db_expander = st.beta_expander(label='CV dashboard:')
        with db_expander:
            available_locations = df3.location.value_counts().sum()
            all_locations = df3.location.value_counts().sum() + df3.location.isnull().sum()
            st.write("**JOB LOCATIONS FROM**", available_locations, "**OF**", all_locations, "**JOBS**")

            
            folium_static(folium_map, width=1250)

            chart1, chart2 = st.beta_columns(2)

            with chart1:
                industry_count = final_jobrecomm.industry.count()
                count_with_null = final_jobrecomm.industry.count() + final_jobrecomm.industry.isnull().sum()
                st.write("**INDUSTRIES PROVIDED FROM**", industry_count, "**OF**", count_with_null, "**JOBS**")
                # fig, ax = plt.subplots()
                # ax = sns.countplot(y=final_jobrecomm['industry'], data=final_jobrecomm, palette="Set3")
                # # ax = px.histogram(final_jobrecomm, x="industry")
                # st.pyplot(fig)

                industry_count = final_jobrecomm.industry.value_counts()
                industry = pd.DataFrame(industry_count)
                industry.reset_index(inplace=True)
                industry.rename({'index': 'Industry', 'industry': 'Count'}, axis=1, inplace=True)
                fig = px.pie(industry, values = "Count", names = "Industry", width=600)
                fig.update_layout(showlegend=True)
                # st.write(fig)
                st.plotly_chart(fig, use_container_width=True)
            
            with chart2:

                final_salary = final_jobrecomm.copy()

                salary_range = []
                for i in final_salary['salary']:
                    x = re.findall('[0-9,]+', str(i))
                    for j in x:
                        salary_range.append(int(j.replace(",",'')))
                        salary_range = [i for i in salary_range if i != 0]
                        salary_range = sorted(salary_range)
                
                salary_df = pd.DataFrame(salary_range, columns=['Salary Range'])
                
                sal_count = salary_df['Salary Range'].count() 
                
                st.write(" **SALARY RANGE FROM**", sal_count, "**SALARY VALUES PROVIDED**")
                
        
                fig2 = px.box(salary_df, y= "Salary Range", width=500)
                fig2.update_yaxes(showticklabels=True)
                fig2.update_xaxes(visible=True, showticklabels=True)
                # st.write(fig2)
                st.plotly_chart(fig2, use_container_width=True)


            
                

        # %%

        # expander for jobs df ---------------------------display#
        db_expander = st.beta_expander(label='Job Recommendations:')

        final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided")

        def make_clickable(link):
            # target _blank to open new window
            # extract clickable text to display for your link
            text = 'more details'
            return f'<a target="_blank" href="{link}">{text}</a>'

        with db_expander:
        #    final1=st.dataframe(final[['title','career level','company','location','industry']].head(no_of_jobs))

    #        def Job_recomm(x):
    #            final_ = final[['title','career level','company','location','industry']]
    #            cl_select = final_[final_["career level"]==x]
    #            return cl_select
    #
    #        result_jd = Job_recomm(CL)
            #st.table(final_jobrecomm.drop(['JobID', "KNN", "TF-IDF", "CV", "webpage","Final"], axis=1))
            final_jobrecomm['webpage'] = final_jobrecomm['webpage'].apply(make_clickable)
            final_jobrecomm['salary'].replace({"0":"Not Available"}, inplace=True)
            final_df=final_jobrecomm[['title','career level','company','location','industry', 'salary', 'webpage']]
            
            [i for i in salary_range if i != 0]
            # link is the column with hyperlinks
            
            show_df = final_df.to_html(escape=False)
            st.write(show_df, unsafe_allow_html=True)
            
            
            
        st.balloons()


 


