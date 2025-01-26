import pandas as pd
import csv 

path_filename = 'gpt-4.csv'

df = pd.read_csv(path_filename, 
                 on_bad_lines='skip',
                 encoding='latin1', 
                 engine='python'  # Use the Python engine
                 )
sample = df.sample(n=1000,random_state=42)
# print(sample)
import re
import matplotlib.pyplot as mb
import numpy as np
import seaborn as sns
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# nlp =spacy.load("en_core_web_sm")
def clean_text(text):
  text = text.lower() #text in lowercase
  text = re.sub(r'\[.*?\]','',text) #remove text in square brackets
  text = re.sub(r'\s+',' ',text) #remove extra spaces
  text = re.sub(r'\w*\d\w*','',text) #remove words containing numbers
  text = re.sub(r'\n','',text) #remove newlines
  text = re.sub(r'[%s]' % re.escape(string.punctuation),'',text) #remove punctuation
  text = re.sub(r'<.*?>+','',text) #remove html tags
  return text

def tokenize_text(text):
  return word_tokenize(text)

def stopwords_removal(text):
  stop_words = set(stopwords.words('english'))
  return [word for word in text if word not in stop_words]

def pos_tagging(text):
  return nltk.pos_tag(text)

def lemmatize_text(text):
  lemmatizer = WordNetLemmatizer()
  return [lemmatizer.lemmatize(word) for word in text]

def preprocess_text(text):
  text = clean_text(text)
  tokens = tokenize_text(text)
  tokens = stopwords_removal(tokens)
  tokens = lemmatize_text(tokens)
  return ' '.join(tokens)

sample['preprocessed_text'] = ""
sample['preprocessed_text'] = sample['conversation'].astype(str).apply(preprocess_text)
# print(sample)
#calculating term frequencies
tf_data= sample["data"].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf_data.columns = ["word","tf"]
tf_data.sort_values("tf",ascending=False)

tf_convo= sample["preprocessed_text"].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf_convo.columns = ["word","tf"]
tf_convo.sort_values("tf",ascending=False)

tf_data[tf_data["tf"] > 1200].plot.bar(x="word",y="tf",color="#483C32")
mb.show()

#Barplot for conversation
tf_convo[tf_convo["tf"] > 900].plot.bar(x="word",y="tf")
mb.show()

#separating doctor patient dialogue for conversation
def separte_convo(conversation):
  if isinstance(conversation, str): # Check if conversation is a string
    doctor_dia = []
    patient_dia =[]
    lines = conversation.split("\n")
    for line in lines:
      if line.startswith("Doctor:"):
        doctor_dia.append(line.replace("Doctor:","").strip())
      elif line.startswith("Patient:"):
        patient_dia.append(line.replace("Patient:","").strip())
    return " ".join(doctor_dia), " ".join(patient_dia)
  else:
    return "", "" # Return empty strings for non-string values

sample[['doctor_conversation','patient_conversation']] = sample['conversation'].apply(lambda x: pd.Series(separte_convo(x)))
print(sample.head())


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sis = SentimentIntensityAnalyzer()

sample["polarity_doctor_convo"] = sample["doctor_conversation"].apply(lambda x: sis.polarity_scores(x)["compound"])
sample["polarity_patient_convo"] = sample["patient_conversation"].apply(lambda x: sis.polarity_scores(x)["compound"])

print(sample["doctor_conversation"])

print(sample["patient_conversation"])

sample["sentiment_label_patient"] = sample["polarity_patient_convo"].apply(lambda x: "positive" if x > 0 else "negative")
sample["sentiment_label_patient"].value_counts()
sentiment_count = sample["sentiment_label_patient"].value_counts()
mb.figure(figsize=(7,7))
mb.pie(sentiment_count,labels=sentiment_count.index,autopct="%1.1f%%",startangle=90)
mb.title("Sentiment Proportion")
mb.show()

from collections import Counter
import re

# Sample of common disease names for extraction (extendable)
common_diseases = [
    'COVID-19', 'diabetes', 'hypertension', 'cancer', 'tuberculosis',
    'asthma', 'stroke', 'malaria', 'influenza', 'hepatitis', 'arthritis',
    'pneumonia', 'HIV', 'Alzheimer', 'Parkinson', 'dengue', 'cholera',
    'epilepsy', 'leukemia', 'depression', 'anxiety', 'migraine', 'eczema',
    'osteoporosis', 'anemia', 'gout', 'obesity', 'cirrhosis', 'bronchitis',
    'lymphoma', 'psoriasis', 'meningitis', 'sinusitis', 'fibromyalgia',
    'sclerosis', 'ulcer', 'glaucoma', 'sepsis', 'schizophrenia', 'bipolar disorder',
    'autism', 'Down syndrome', 'lupus', 'sarcoidosis', 'tetanus', 'rabies',
    'measles', 'mumps', 'rubella', 'pertussis', 'diphtheria', 'polio',
    'smallpox', 'Ebola', 'Zika virus', 'yellow fever', 'encephalitis',
    'lyme disease', 'tinnitus', 'vertigo', 'vitiligo', 'scabies', 'ringworm',
    'conjunctivitis', 'keratitis', 'otitis', 'colitis', 'IBS', 'Celiac disease',
    'Crohn’s disease', 'diverticulitis', 'pancreatitis', 'renal failure',
    'urinary tract infection', 'prostatitis', 'endometriosis', 'fibroid',
    'cystitis', 'infertility', 'ovarian cyst', 'PCOS', 'impetigo',
    'cellulitis', 'gangrene', 'necrosis', 'alopecia', 'sickle cell anemia',
    'thalassemia', 'hemophilia', 'varicose veins', 'hemorrhoids',
    'carpal tunnel syndrome', 'tendinitis', 'plantar fasciitis',
    'herniated disc', 'sciatica', 'Bell’s palsy', 'Guillain-Barre syndrome',
    'myasthenia gravis', 'dystonia', 'sleep apnea'
]

#function to extract diseases from text using a predefined list
def extract_diseases(text,dis_list):
  found_diseases = []
  for disease in dis_list:
    if re.search(rf'\b{re.escape(disease)}\b', text, re.IGNORECASE):
      found_diseases.append(disease)
  return found_diseases

sample['extracted_diseases'] = sample['data'].apply(lambda x: extract_diseases(str(x),common_diseases))
# print(sample)

all_diseases = [diseases for diseases in sample['extracted_diseases'] for diseases in diseases]
disease_count = Counter(all_diseases)
Most_repeated_diseases = disease_count.most_common(10)
print("The Top 10 Most Repeated diseases are:")
for disease,count in Most_repeated_diseases:
  print(f"{disease}: {count} occurences")
  
disease_counts = Most_repeated_diseases
mb.figure(figsize=(12,8))
disease_labels = [disease for disease, count in disease_counts]
count_values = [count for disease, count in disease_counts]
x_values = range(len(disease_labels))

mb.fill_between(x_values, count_values, color = "Red")
mb.title("Most Repeated Diseases")
mb.xticks(x_values, disease_labels, rotation='vertical')
mb.xlabel("Disease")
mb.ylabel("Occurrences")
mb.tight_layout()
mb.show()

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

def search_disease():
    disease_name = disease_entry.get()
    if disease_name:
        results_text.delete("1.0", tk.END)
        if disease_name in common_diseases:
            disease_data = sample[sample['extracted_diseases'].apply(lambda x: disease_name in x)]
            if not disease_data.empty:
                num_occurrences = len(disease_data)
                results_text.insert(tk.END, f"Number of occurrences for {disease_name}: {num_occurrences}\n\n")
                
                image_path = f'images/{disease_name}.jpg'
                if os.path.exists(image_path):
                  try:
                    image = Image.open(image_path)
                    photo = ImageTk.PhotoImage(image)
                    image_label = tk.Label(window, image=photo)
                    image_label.image = photo #Keep a reference
                    image_label.grid(row=3, column = 0, columnspan=3, padx=7, pady=6)

                    results_text.insert(tk.END, f"Image displayed \n")
                  except Exception as e:
                    print("Error displaying image",e)
                else:
                    results_text.insert(tk.END, f"Image not found at {image_path}\n")

                for index, row in disease_data.iterrows():
                    results_text.insert(tk.END, f"\nConversation:\n{row['conversation']}\n")

            else:
                results_text.insert(tk.END, f"No data found for {disease_name}")
        else:
            results_text.insert(tk.END, f"{disease_name} is not in the list of common diseases.")
    else:
        results_text.insert(tk.END, "Please enter a disease name.")
window = tk.Tk()
window.title("Disease Search")
disease_label = ttk.Label(window, text="Enter Disease:")
disease_label.grid(row=0, column=0, padx=5, pady=5)

disease_entry = ttk.Entry(window)
disease_entry.grid(row=0, column=1, padx=5, pady=5)

search_button = ttk.Button(window, text="Search", command=search_disease)
search_button.grid(row=0, column=2, padx=5, pady=5)

results_label = ttk.Label(window, text="Results:")
results_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)


results_text = tk.Text(window, wrap=tk.WORD)
results_text.grid(row=2, column=0, columnspan=3, padx=5, pady=5)


# Run the GUI
window.mainloop()

import requests 
import json
from datetime import datetime, timezone

start_time =datetime(2025, 6, 12, 5, 30, 0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") #for every instance change the date and time
url = "https://api.cal.com/v2/bookings"

payload = json.dumps({
    "start": start_time,  # Replace with your desired date/time in UTC
    "eventTypeId": 1715767,  # Replace with your event type ID from Cal.com
    "attendee": {
        "name": "Surya",  # Attendee's name
        "email": "msurya200311@gmail.com",  # Attendee's email
        "timeZone": "Asia/Kolkata",  # Attendee's time zone
        "language": "en"  # Preferred language
    },
    "guests": [
        "214M1A3133@vemu.org"  # Additional guest email(s), if any
    ]
})
# API Authorization and Headers
headers = {
    'Authorization': 'Bearer cal_live_4e2f37409c7cd4660808a842e97be7b7',  # Replace with your Cal.com API key
    'Content-Type': 'application/json',
    'cal-api-version': '2024-08-13'
}
try:
    # Make the POST request to schedule the booking
    response = requests.post(url, headers=headers, data=payload)

    # Handle the API response
    if response.status_code == 201:  # HTTP 201 indicates booking was successfully created
        print("Booking created successfully!")
        print(json.dumps(response.json(), indent=4))  # Pretty-print the response JSON
    else:
        print("Failed to create booking. Error:")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=4))

except requests.exceptions.RequestException as e:
    print(f"Error occurred during the API request: {e}")