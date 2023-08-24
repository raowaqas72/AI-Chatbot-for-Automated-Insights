from flask import Flask, render_template, request, send_file
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import pandas as pd
import plotly.express as px
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        print("probability", str(r[1]))
        print("intent", classes[r[0]])
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list

def get_response(intents_list, intents_json):
    global plot_url
    tag = intents_list[0]['intent']
    print(tag)
    # list_of_intents = intents_list['intents']
    #print(list_of_intents)
    #for i in list_of_intents:
        # if i['tag'] == tag:
    #     result = random.choice(i['responses'])
    
    if tag == "Age_Distribution_by_City":
        # Data processing and visualization code here
        result = "Age distribution visualization generated."
        data = pd.read_csv('uae_city_data_rao.csv', encoding='ISO-8859-1')

        # Group data by city and calculate age group distribution
        age_groups = ['Age 0-18', 'Age 19-35', 'Age 36-50', 'Age 51+']
        age_distribution_by_city = data.groupby('City')[age_groups].sum()

        # Create a stacked bar chart using Plotly
        fig = px.bar(age_distribution_by_city,
                    x=age_distribution_by_city.index,
                    y=age_groups,
                    title='Age Distribution by City',
                    labels={'x': 'City', 'y': 'Population'},
                    width=800,
                    height=600,
                    barmode='stack')

        # Export the Plotly graph to an interactive HTML file
        fig.write_html("age_distribution_plotly.html")
        plot_url = "http://127.0.0.1:5003/get_age_distribution_plot"
        return plot_url        

    elif tag == "totalpopulationbycity":
        # Data processing and visualization code here
        result = "Gender population visualization generated."
    elif tag == "AverageHealthcare":
        # Data processing and visualization code here
        result = "Average healthcare expenditure visualization generated."
    elif tag == "temperaturevsmortality":
        # Data processing and visualization code here
        result = "Temperature vs. mortality visualization generated."
#        print(i['tag'])    #break
    return result

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the trained model
model = load_model('chatbot_model_v1.h5')

# Load the preprocessed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Load the intents JSON data
intents_file = open('intents.json', errors="ignore").read()
intents_json = json.loads(intents_file)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    intents = predict_class(user_input, model)
    response = get_response(intents, intents_json)
    return response

if __name__ == '__main__':
    app.run(debug=True)
