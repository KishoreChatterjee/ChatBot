import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\Kishore Chatterjee\\Desktop\\chatbot python\\chatbot\\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list


def get_word_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice (i['responses'])
#             break
#     return result

def get_response(intents_list, intents_json, message):
    print("Intents List:", intents_list)

    if intents_list:
        top_intent = intents_list[0]['intent']
        probability = intents_list[0]['probability']
        print(f"Top Intent: {top_intent}, Probability: {probability}")

        tag_found = False
        list_of_intents = intents_json['intents']

        result = "I'm sorry, I don't understand that."  # Default response

        for i in list_of_intents:
            if i['tag'] == top_intent:
                tag_found = True
                result = random.choice(i['responses'])
                break

        if not tag_found:
            # Query WordNet for synonyms
            synonyms = get_word_synonyms(message)
            if synonyms:
                result = f"Here are some synonyms: {', '.join(synonyms)}"

        return result
    else:
        return "I'm sorry, I don't understand that."

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents,message)
    print (res)
    