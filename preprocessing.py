import pandas as pd
import enchant
import base64
import nltk
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz


def initialize(df, predefined_dataset):

    city_state_sample = []
    for city in range(0, len(df)):
        sample_city = df['city'][city]
        sample_state = df['state'][city]

        string_output = str(sample_city)+','+str(sample_state)
        city_state_sample.append(string_output)

    city_state_predefined = predefined_dataset['city_state']
    city_state_predefined = list(city_state_predefined)

    return city_state_sample, city_state_predefined


def preprocess(df, predefined_dataset, city_state_predefined):

    sample_cities = []
    sample_city = df['city']
    sample_city = list(sample_city)
    predefined_city = predefined_dataset['city']
    predefined_city = list(predefined_city)

    for city in sample_city:
        if city not in predefined_city:
            sample_cities.append(city)
    sample_states = []

    sample_state = df['state']
    sample_state = list(sample_state)

    predefined_state = predefined_dataset['state_id']
    predefined_state = list(predefined_state)

    for state in sample_state:
        if state not in predefined_state:
            sample_states.append(state)

    not_present_city_state = []

    for city in range(0, len(sample_city)):

        if sample_city[city] in sample_cities:
            string1 = sample_city[city]+','+sample_state[city]
            not_present_city_state.append(string1)

    for state in range(0, len(sample_state)):

        if sample_state[state] in sample_states:
            string1 = sample_city[state]+','+sample_state[state]
            not_present_city_state.append(string1)

    not_present_city_state = set(not_present_city_state)

    cities_corrected = {}
    minimum = 999
    for iterator_city_state in not_present_city_state:
        minimum = 999
        for correct_city_state in city_state_predefined:
            temp = enchant.utils.levenshtein(
                iterator_city_state, correct_city_state)
            if temp < 4:
                if temp < minimum:
                    minimum = temp
                    cities_corrected[iterator_city_state] = correct_city_state

            else:
                continue

    corrected_dataframe = pd.DataFrame()
    corrected_dataframe['Original'] = cities_corrected.keys()
    corrected_dataframe['Changed'] = cities_corrected.values()

    return corrected_dataframe


def preprocess_abbreviation(df):

    abbreviation = pd.read_excel('Dataset/Abrreviations.xlsx')
    proper_abbreviations = abbreviation['Abbreviations']
    proper_abbreviations = list(proper_abbreviations)

    correct_abbreviations = abbreviation['Corrected_Abbreviations']
    correct_abbreviations = list(correct_abbreviations)
    mapping_dict = {}
    for something in range(0, len(proper_abbreviations)):
        mapping_dict[proper_abbreviations[something]
                     ] = correct_abbreviations[something]

    Address = df['address']
    Address = list(Address)
    tokens = []
    for part in Address:
        nltk_tokens = nltk.word_tokenize(part)
        tokens.append(nltk_tokens)

    for part in tokens:
        for some in range(0, len(part)):
            if part[some] in mapping_dict.keys():
                part[some] = mapping_dict[part[some]]

    for tok in range(0, len(tokens)):
        tokens[tok] = ' '.join(tokens[tok])

    return tokens


def update_values(response, dataframe):

    original_not_to_change = []
    changed = []
    state = list(dataframe['state'])
    city = list(dataframe['city'])

    for index in response['selected_rows']:
        original_not_to_change.append(index['Original'])

    original = list(response['data']['Original'])
    changed = list(response['data']['Changed'])

    for part in range(0, len(original)):
        parted_original = original[part].split(',')
        parted_changed = changed[part].split(',')
        if original[part] in original_not_to_change:
            continue
        elif parted_original[0] in city:
            ind = city.index(parted_original[0])
            city[ind] = parted_changed[0]
            state[ind] = parted_changed[1]

    dataframe['city'] = city
    dataframe['state'] = state

    return dataframe


def filedownload(df, decide):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    if decide == 1:
        href = f'<a href = "data:file/csv;base64,{b64}" download="Address.csv">Download CSV File</a>'
    else:
        href = f'<a href = "data:file/csv;base64,{b64}" download="Provider.csv">Download CSV File</a>'

    return href

# ---------------------------------------------------------------------------------------------------
# Provider Functions


def checkAutoCorrect(tokenlist):
    '''This function checks whether words are english or not and corrects the spelling mistakes
    Input : list of token
    Output : corrected string
    '''
    temp = ''
    spell = SpellChecker(language='en')
    for token in tokenlist:
        # check for abbreviation
        if len(token) <= 3:
            temp = temp+" "+token
        # check if word is english
        elif dict.check(token):
            temp = temp+" "+token
        else:
            # correct the word if spelling mistakes
            correctword = str(spell.correction(
                token)) if spell.correction(token) else token
            temp = temp+" "+correctword

    original = ' '.join(tokenlist)
    print(
        f'original string:{original} , replaced string:{temp}, ratio: {fuzz.WRatio(original, temp)}')

    return temp.upper()


def update_provider_names(response, dataframe):

    original_not_to_change = []
    changed = []

    provider = list(dataframe['Provider_Name'])

    for index in response['selected_rows']:
        original_not_to_change.append(index['Provider_Name'])

    original = list(response['data']['Provider_Name'])
    changed = list(response['data']['Filtered_Name'])

    for part in range(0, len(original)):
        parted_original = original[part]
        parted_changed = changed[part]
        if original[part] in original_not_to_change:
            continue
        elif parted_original in provider:
            ind = provider.index(parted_original)
            provider[ind] = parted_changed

    dataframe['Provider_Name'] = provider
    return dataframe
