# Job-and-Talent-Recommendation-System

<!-- Table of Content
<br>
[Aim](#Aim)
<br>
[Business Values](#Goal)
<br>
[Process](#Process)
<br>
[Data Collection](#DataCollection)
<br>
[OCR (Optical Character Recognition)](#OCR)
<br>
[NLP (Natural Language Processing)](#NLP)
<br>
[How does it work?](#Method)
<br>
[Recommendations](#Recommendations)
<br>

<a name="Aim"/>
<a name="Goal"/>
<a name="Process"/>
<a name="DataCollection"/>
<a name="OCR"/>
<a name="NLP"/>
<a name="Method"/>
<a name="Recommendations"/>
 -->

<h3>Aim</h3>
Assist applicants to search for potential jobs
<br>
Assist recruiters to find talents

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Business Values</h3>
Jobmigo brings the job world to you
<br>
- Simple to use: Drag and Drop
<br>
- Save time 
<br>
- Tailor-made dashboard
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Process</h3>
<img width="700" alt="Screenshot 2021-08-10 at 14 08 38" src="https://user-images.githubusercontent.com/80112729/128816859-87e061da-b9e1-4880-ba1a-6e38122ff412.png">
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Data Collection</h3>
<!-- - Web Scraping  -->
Web Scraping 
<img width="700" alt="Screenshot 2021-08-10 at 14 11 29" src="https://user-images.githubusercontent.com/80112729/128817149-7059ea2a-9baa-437a-a702-f18d2a900025.png">
Scraping useful information from websites, including Title, Company, Location, Salary and Post Date etc. 
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- - Data Preprocessing -->
<br>
Data Preprocessing
<img width="700" alt="Screenshot 2021-08-10 at 14 11 47" src="https://user-images.githubusercontent.com/80112729/128817186-7571f83e-8346-48a4-91fc-6be351b35625.png">
Extract useful content from scraped data. E.g. HK$ 11,000 - HK$ 25, 000/month to 11000, 25000. 
<br>
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


1. Job posts: 
<br>
20,137 jobs
<br>
Across 5 main industries 
<br>
Method: Web scraping from Jobsdb
<br>
<br>
2. CV: 
<br>
54 CVs
<br>
From friends & google search

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>OCR (Optical Character Recognition)</h3>
<img width="700" alt="Screenshot 2021-08-03 at 18 29 29" src="https://user-images.githubusercontent.com/80112729/128001196-924fb26f-e449-429d-b2ee-a1221a827878.png">
<br>

<!-- <br> -->
Convert a pdf file of text document into a machine encoded text. In other words, it can be further processed as a string format in jupyter notebook.
<br>

<img width="700" alt="Code" src="https://user-images.githubusercontent.com/80112729/128001347-04f6e853-50f1-4e6e-aa49-ec12b8494d0d.png">
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>NLP (Natural Language Processing)</h3>
1.Word tokenization:
<br>
- Word tokenization
<img width="700" alt="Word_To" src="https://user-images.githubusercontent.com/80112729/128820414-9106bfd3-f760-4c7a-894c-2f2ac3cb35b6.png">
<br>

2.Stop words removal 
<img width="1210" alt="Stopwords_removeal" src="https://user-images.githubusercontent.com/80112729/128821421-e20a99f3-44b1-49a5-b619-d233449ba5cf.png">
<br>

3.Lemmatization
Converting a word to its base form. 
<br>
"motivated" to "motivate", "Learning" to "Learn".
<br>
<br>
4.Bigram Collection Finder
<br>
Finding meaning double words.
<br>
<img width="428" alt="Bi_gram" src="https://user-images.githubusercontent.com/80112729/128822261-6124d4a7-a9c6-499a-9c26-adbe516707a4.png">

<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>How does it work?</h3>
<br>
<img width="700" alt="Recommendatikon system" src="https://user-images.githubusercontent.com/80112729/128826802-9368a2e6-c14b-4645-a213-56b8041a65ab.png">
<br>
<br>
CountVectorizer
<img width="700" alt="CV" src="https://user-images.githubusercontent.com/80112729/128827727-a5f9febf-4599-4c7a-823e-7c4b741405c8.png">


<br>
TF-IDF
<img width="700" alt="TF-IDF" src="https://user-images.githubusercontent.com/80112729/128827746-b2a9ed56-828c-4a29-86bf-248e4d29e3e9.png">


<br>
KNN
<img width="700" alt="KNN" src="https://user-images.githubusercontent.com/80112729/128827771-f4bbce47-1fb0-4536-ac61-b97c0f66eefb.png">


<br>
Cosine Similarity 
<img width="900" alt="CosSim" src="https://user-images.githubusercontent.com/80112729/128829905-4ae21f09-73a4-4f3d-bacb-b936b431e85c.png">

<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h3>Recommendations</h3>
<img width="700" alt="Recommendationssss" src="https://user-images.githubusercontent.com/80112729/128827831-e5ae55cd-394d-4ac5-8522-95bdf8a5bb8a.png">


<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
Streamlit Demo


https://user-images.githubusercontent.com/80112729/128828052-0381a96c-d994-4499-908b-ef59f5f27aa1.mp4



