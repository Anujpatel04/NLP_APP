from flask import Flask
import mysql.connector
from flask import render_template,request
import re
from nltk.corpus import stopwords
from nltk.util import bigrams,trigrams
from nltk import word_tokenize
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer


app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():   
    return render_template('index.html')
#-----------------------------------------------------------------
@app.route('/new',methods=['GET','POST'])
def new():
    bg=[]
    if request.method=='POST':
        data=request.form['input']
        words = word_tokenize(data)
        bg=list(bigrams(words))
    return render_template('new.html',bg=bg)
#---------------------------------------------------------------
@app.route('/trigram', methods=['GET', 'POST'])
def trigram():
    tri = []
    if request.method == 'POST':
        data = request.form['input']
        words = word_tokenize(data)
        # Generate trigrams with the last word of the current trigram repeated in the next
        for i in range(0, len(words) - 2, 2):  # Increment by 2 to skip words
            if i + 2 < len(words):
                tri.append((words[i], words[i + 1], words[i + 2]))
    return render_template('trigram.html', tri=tri)
#--------------------------------------------------------------
@app.route('/wtoken',methods=['GET','POST'])
def wtoken():
    words=[]
    if request.method=='POST':
        input=request.form['input']
        words=list(word_tokenize(input))
    return render_template('wtoken.html',token=words)
#------------------------------------------------------------
@app.route('/stoken',methods=['GET','POST'])
def stoken():
    s_token=[]
    if request.method=='POST':
        input=request.form['input']
        s_token=list(sent_tokenize(input))
    return render_template('stoken.html',token=s_token)
#-------------------------------------------------------------
@app.route('/swords', methods=['GET', 'POST'])
def swords():
    sp = []  
    if request.method == 'POST':
        sp = stopwords.words('english') 
    return render_template('swords.html', sp=sp)
#-------------------------------------------------------------
@app.route('/tfvector', methods=['GET', 'POST'])
def tfvector():
    if request.method == 'POST':
        input_text = request.form['input']
        
        # Initialize stemmer and lemmatizer
        ps = PorterStemmer()
        wordnet = WordNetLemmatizer()
        sentences = sent_tokenize(input_text)

        corpus = []

        # Preprocess the input text and create a cleaned corpus
        for i in range(len(sentences)):
            review = re.sub('[^a-zA-Z]', ' ', sentences[i])  # Remove non-alphabetical characters
            review = review.lower()
            review = review.split()
            review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]  # Lemmatization
            review = ' '.join(review)
            corpus.append(review)

        # Create Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        X_bow = cv.fit_transform(corpus).toarray()  # Bag of Words representation
        vocab = cv.get_feature_names_out()  # Get the words for the columns

        # Create TF-IDF model
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer()
        X_tf = tf.fit_transform(corpus).toarray()  # TF-IDF representation

        # Convert arrays to lists for easier handling in Jinja
        X_bow = X_bow.tolist()
        X_tf = X_tf.tolist()

        # Pass the data to the template
        return render_template('tfvector.html', X_bow=X_bow, X_tf=X_tf, vocab=vocab, corpus=corpus)



if __name__=='__main__':
    app.run()