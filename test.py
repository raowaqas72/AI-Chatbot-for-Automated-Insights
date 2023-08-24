import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbot_model_v1.h5')

# Load the preprocessed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

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
    ERROR_THRESHOLD = 0.25  # Adjust this threshold based on your model's performance
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            print(i['tag'])
            result = random.choice(i['responses'])
            if i['tag']=="Age_Distribution_by_City":
                
            # data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')
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

                # # Group data by city and calculate age group distribution
                # age_groups = ['Age 0-18', 'Age 19-35', 'Age 36-50', 'Age 51+']
                # age_distribution_by_city = data.groupby('City')[age_groups].sum()

                # # Create a stacked bar chart
                # plt.figure(figsize=(10, 6))
                # age_distribution_by_city.plot(kind='bar', stacked=True)
                # plt.title('Age Distribution by City')
                # plt.xlabel('City')
                # plt.ylabel('Population')
                # plt.xticks(rotation=45)
                # plt.tight_layout()

                # # Show the plot
                # plt.legend(title='Age Groups')
                # plt.show()
            if i['tag']=="totalpopulationbycity":
                
                data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')
                # Group data by city and calculate male and female populations
                gender_population_by_city = data.groupby('City')[['Male Population', 'Female Population']].sum().reset_index()

                # Create a grouped bar chart using Plotly
                fig = px.bar(gender_population_by_city,
                            x='City',
                            y=['Male Population', 'Female Population'],
                            title='Gender Population by City',
                            labels={'City': 'City', 'value': 'Population'},
                            width=800,
                            height=600,
                            barmode='group')

                # Export the Plotly graph to an interactive HTML file
                fig.write_html("gender_population_plotly.html")

                if i['tag']=="AverageHealthcare":
                    data = pd.read_csv('uae_city_data_rao.csv', encoding='ISO-8859-1')

                    # Group data by city and calculate average healthcare expenditure
                    average_expenditure_by_city = data.groupby('City')['Healthcare Expenditure (% of Income)'].mean().reset_index()

                    # Create a line chart using Plotly
                    fig = px.line(average_expenditure_by_city,
                                x='City',
                                y='Healthcare Expenditure (% of Income)',
                                title='Average Healthcare Expenditure by City',
                                labels={'City': 'City', 'Healthcare Expenditure (% of Income)': 'Healthcare Expenditure (% of Income)'},
                                width=800,
                                height=600,
                                markers=True)

                    # Export the Plotly graph to an interactive HTML file
                    fig.write_html("average_expenditure_plotly.html")

            if i['tag']=="temperaturevsmortality":
                # Load the data from the CSV file
                # Read the CSV data
                data = pd.read_csv('uae_city_data_rao.csv', encoding='ISO-8859-1')

                # Create a scatter plot using Plotly
                fig = px.scatter(data,
                                x='Average Temperature (°C)',
                                y='Mortality Rate (%)',
                                title='Temperature vs. Mortality Rate',
                                labels={'Average Temperature (°C)': 'Average Temperature (°C)', 'Mortality Rate (%)': 'Mortality Rate (%)'},
                                width=800,
                                height=600,
                                color_discrete_sequence=['blue'])

                # Export the Plotly graph to an interactive HTML file
                fig.write_html("temperature_mortality_plotly.html")



            break
    return result

# Load the intents JSON data
intents_file = open('intents.json', errors="ignore").read()
intents_json = json.loads(intents_file)

# User input
user_input = input("what you want: ")

# Get the predicted intent
intents = predict_class(user_input, model)

# Get the response
response = get_response(intents, intents_json)

print("Bot response:", response)
