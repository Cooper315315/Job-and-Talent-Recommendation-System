# Job-and-Talent-Recommendation-System

## Table of contents
* [Aim](#aim)
* [Business Values](#business-values)
* [Process](#process)
* [Data Collection](#data-collection)
* [Data Preprocessing](#data-preprocessing)
* [Optical Character Recognition](#optical-character-recognition)
* [Natural Language Processing](#natural-language-processing)
* [Recommendations](#recommendations)
* [Result](#result)
* [Conclusion](#conclusion)
* [Next Step](#next-step)

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

For job seeker:
Firstly, Input the CV and then it will be processed by OCR. After that, the CV and job posts will undergo NLP process. And then both CV and job post will be compared by varies methods to find the similarity. Lastly, the system will list out recomendation of jobs.
<br>
<br>
For recruiter:
Firstly, Input the job post and it will undergo NLP process along with CV in database. And then both CV and job post will be compared by varies methods to find the similarity. Lastly, the system will list out recomendation of candidates.

<img width="700" alt="Screenshot 2021-08-10 at 14 08 38" src="https://user-images.githubusercontent.com/80112729/128816859-87e061da-b9e1-4880-ba1a-6e38122ff412.png">
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Data Collection</h3>
Web Scraping 
Scraping useful information from websites, including Title, Company, Location, Salary and Post Date etc. 
<img width="700" alt="Screenshot 2021-08-10 at 14 11 29" src="https://user-images.githubusercontent.com/80112729/128817149-7059ea2a-9baa-437a-a702-f18d2a900025.png">

<h3>Data Preprocessing</h3>
Extract useful content from scraped data. E.g. HK$ 11,000 - HK$ 25, 000/month to 11000, 25000. 
<img width="700" alt="Screenshot 2021-08-10 at 14 11 47" src="https://user-images.githubusercontent.com/80112729/128817186-7571f83e-8346-48a4-91fc-6be351b35625.png">
<br>

1. Job:
* 20,137 jobs
* Across 5 main industries (including main industry such as finance, banking, logistics, and IT) 
* Method: Web scraping from Jobsdb

2. CV: 
* 54 CVs
* From friends & google search

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Optical Character Recognition</h3>
Convert a pdf file of text document into a machine encoded text. In other words, it can be further processed as a string format in jupyter notebook.
<br>

<img width="700" alt="Screenshot 2021-08-03 at 18 29 29" src="https://user-images.githubusercontent.com/80112729/128001196-924fb26f-e449-429d-b2ee-a1221a827878.png">
<br>

<img width="700" alt="Code" src="https://user-images.githubusercontent.com/80112729/128001347-04f6e853-50f1-4e6e-aa49-ec12b8494d0d.png">
<br>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Natural Language Processing</h3>
1.Word tokenization:
<br>
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

<h3>Recommendations</h3>
<img width="700" alt="Recommendatikon system" src="https://user-images.githubusercontent.com/80112729/128826802-9368a2e6-c14b-4645-a213-56b8041a65ab.png">
<br>
<br>
CountVectorizer
<br>
- It converts a set of strings into a sparse matrix 
- One hot encode the text document
<br>
<img width="700" alt="CV1" src="https://user-images.githubusercontent.com/80112729/129741412-f7d75f35-5df3-4d76-8c57-c546ba118ebf.png">
<br>

TF-IDF (Term Frequency-Inverse Document Frequency)
<br>
- Term Frequency, TF: The number of times a term occurs in a document
- Inverse Document Frequency, IDF: It measures how important a term is
<br>
<img width="700" alt="TF-IDF1" src="https://user-images.githubusercontent.com/80112729/129742243-e2b4b81a-d5b7-41b5-89fa-12d504edd1b7.png">
(Please note that for the word "earth", the IDF should be log(2/1)=0.3)
<br>
<br>
<br>
KNN
<br>
<img width="700" alt="KNN1" src="https://user-images.githubusercontent.com/80112729/129743390-42f8bdb2-a982-4391-8470-09f3bcdadce5.png">
<br>

Cosine Similarity 
<img width="900" alt="CosSim" src="https://user-images.githubusercontent.com/80112729/128829905-4ae21f09-73a4-4f3d-bacb-b936b431e85c.png">
<br>
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h3>Result</h3>
<img width="700" alt="Recommendationssss" src="https://user-images.githubusercontent.com/80112729/128827831-e5ae55cd-394d-4ac5-8522-95bdf8a5bb8a.png">
<br>
Streamlit Demo
<br>

https://user-images.githubusercontent.com/80112729/128828052-0381a96c-d994-4499-908b-ef59f5f27aa1.mp4

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<h3>Conclusion</h3>
Jobmigo provides a robust recommendation platform for both job seekers and recruiters. The goal is to simplify the job seeking and candidate screening process, this can be achieved with Jobmigo. It is easy to use (just upload and run). For recruiters, it saves time screening CV manually and provides a less bias result; and for job seekers, it provides a tailor made and interactive dashboard with relevant job opportunities. 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h3>Next Step</h3>
* Further develop recommendation system to return accurate relevant results
<br>
* Engage with job posting companies to gain access to database / API
<br>
* Invite applicants to join the CV database
<br>
* More features on the dashboard
