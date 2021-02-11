import streamlit as st
import json as j
import pickle
import pyaudio, wave
import pandas as pd
import speech_recognition as sr
import os
import re
import numpy as np
import string
import nltk
from sound import sound
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Moody: Meeting Feedback")
    #Insert an image
    image = Image.open(os.path.join(os.getcwd(), 'speech.jpg'))
    st.image(image, use_column_width=True)
    st.header("Recorded Audio Sentiment Analysis")
    st.write("Please insert your audio speech")

    # Train the model
    if st.button('Train'):
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = preprocessing()
        model, acc = train(X_train, X_test, y_train, y_test)
        save_model(model)
        st.write("Training completed")
        st.write("The training accuracy is " + acc)

    # Record the audio
    if st.button('Record'):
        with st.spinner('Recording for 5 seconds ....'):
            sound.record()
        st.success("Recording completed")
    
    loaded_model = load_model('model.sav')
    # Play the audio
    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open("recorded.wav", 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")
        transcript = readtranscribe("recorded.wav")
        st.write(transcript)
    
    # Predict the model
    if st.button('Predict'):
        transcript = readtranscribe("recorded.wav")
        st.write(predict(loaded_model, transcript))
        st.write("All prediction has been made")

    # Uploaded recording file
    st.header("Uploaded Recording File Sentiment Analysis")
    uploaded_file = st.file_uploader("Choose an audio...", type="wav")
    if uploaded_file is not None:
        text = readtranscribe(uploaded_file)
        st.write(predict(loaded_model, text))

    # Feedback on meeting
    st.header("Chat Room Sentiment Analysis")
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        st.write(predict(loaded_model, sentence))

    # Meeting minutes summarizer
    st.header("Meeting Minutes Summarizer")
    body = st.text_input('Input your text here:')
    if body:
        st.write("The summary will be:")
        # 1 Create the word frequency table
        freq_table = _create_frequency_table(body)

        # 2 Tokenize the sentences
        sentences = sent_tokenize(body)

        # 3 Important Algorithm: score the sentences
        sentence_scores = _score_sentences(sentences, freq_table)
    
        # 4 Find the threshold
        threshold = _find_average_score(sentence_scores)

        # 5 Important Algorithm: Generate the summary
        summary = _generate_summary(sentences, sentence_scores, threshold)

        st.write(summary)

def readtranscribe(path):
    r = sr.Recognizer()
    wavfile = sr.AudioFile(path)
    with wavfile as source:
        audio = r.record(source)
        value = r.recognize_google(audio)
        return value

def _create_frequency_table(text_string) -> dict:

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def preprocessing():
    fileName = os.path.join(os.getcwd(), "train.csv")
    twitterText = pd.read_csv(fileName)
    for i,emotion in enumerate(list(twitterText.Emotion.unique())):   
        twitterText["Emotion"]= twitterText["Emotion"].replace(emotion, i)
    
    words = stopwords.words("english")

    # Remove user with @
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt 
    twitterText['Text'] = twitterText['Text'].apply(lambda x: remove_pattern(x, "@[\w]*"))

    # Remove punctuation
    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    round1 = lambda x: clean_text_round1(x)
    twitterText['Text'] = twitterText['Text'].apply(round1)

    # Get rid of additional punctuation if needed
    def clean_text_round2(text):
        '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return text

    round2 = lambda x: clean_text_round2(x)
    twitterText['Text'] = twitterText['Text'].apply(round2)

    # Remove the stopwords
    stop = stopwords.words('english')
    twitterText['Text'] = twitterText['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Perform stemming and lemmatization
    lem = WordNetLemmatizer()
    twitterText['Text'] = twitterText['Text'].apply(lambda x: " ".join(lem.lemmatize(x) for x in x.split()))

    stemmer = SnowballStemmer('english')
    twitterText['Text'] = twitterText['Text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    X_train, X_test, y_train, y_test = train_test_split(twitterText['Text'], twitterText.Emotion, test_size=0.2)

    return X_train, X_test, y_train, y_test

def train(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                        ('chi',  SelectKBest(chi2, k=10000)),
                        ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

    model = pipeline.fit(X_train, y_train)

    # vectorizer = model.named_steps['vect']
    # chi = model.named_steps['chi']
    # clf = model.named_steps['clf']

    # feature_names = vectorizer.get_feature_names()
    # feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    # feature_names = np.asarray(feature_names)

    # target_names = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
    # print("top 10 keywords per class:")
    # for i, label in enumerate(target_names):
    #     top10 = np.argsort(clf.coef_[i])[-10:]
    #     print("%s: %s" % (label, " ".join(feature_names[top10])))

    acc = str(model.score(X_test, y_test))
    return model, acc

# Saving the model
def save_model(model):
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def predict(model, sentence):
    result_dict={0:'sadness',1:'anger',2:'love',3:'surprise',4:'fear',5:'joy'}
    text_list = [sentence]
    result = model.predict(text_list)
    result = result.tolist()
    final_result = []
    for i in result:
        final_result.append(result_dict[i])
    
    sentences = []
    for m in range(len(text_list)):
        res = "The text is " + text_list[m] + " and the emotion is " + final_result[m]
        sentences.append(res)
    return sentences[0]


if __name__ == "__main__":
    main()
