import pathlib
import streamlit as st
import os
import nltk
import pandas as pd
import enchant
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz

Cluster = {}
threshold = 0.60
length = 1
count = 0
flag = 0


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
# -------------------------------------------------------------------------------------------------------------------
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
    provider = list(dataframe['Provider Name"'])

    for index in response['selected_rows']:
        original_not_to_change.append(index['Provider Name'])

    original = list(response['data']['Provider Name'])
    changed = list(response['data']['Filtered Rows'])

    for part in range(0, len(original)):
        parted_original = original[part]
        parted_changed = changed[part]
        if original[part] in original_not_to_change:
            continue
        elif parted_original in provider:
            ind = provider.index(parted_original)
            provider[ind] = parted_changed

    dataframe['Provider Name'] = provider
    return dataframe


# MAIN FUNCTION:
# ----------------------------------------------------

st.set_page_config(layout="wide")

with st.sidebar:
    st.markdown("""
    __Select an entity from list below__
    """)
    page = st.selectbox('Select:', ['Address Issue', 'Provider Issue'])

if page == 'Address Issue':

    st.write(""" # Address Data Quality Issue:
    Please upload the required dataset. """)

    uploaded_file = st.sidebar.file_uploader(
        "Choose a Excel or CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:

        if 'input_df' not in st.session_state:

            st.session_state['input_df'] = pd.read_csv(uploaded_file)
            st.write("File Preview:")
            st.dataframe(st.session_state['input_df'], width=650, height=550)

        if st.sidebar.button(' Preprocess ↻ '):

            if 'preprocess' not in st.session_state:
                st.session_state['preprocess'] = True
                st.session_state['abbrev'] = True
                st.session_state['final'] = True

    if 'final' in st.session_state:

        if st.sidebar.button('Generate Complete File..'):

            if st.session_state['abbrev'] == True:
                data_frame = update_values(
                    st.session_state['response'], st.session_state['dataframe'])
                st.session_state['dataframe'] = data_frame
                tokens = preprocess_abbreviation(data_frame)
                st.session_state['dataframe']['address'] = tokens
                st.session_state['abbrev'] = False

            st.session_state['preprocess'] = False
            st.markdown("""
                [Download CSV file](st.session_state['dataframe'])
            """)
            st.write('#### Complete Preprocessed File: ')
            st.dataframe(st.session_state['dataframe'], width=650, height=550)

    if 'preprocess' in st.session_state:

        if st.session_state['preprocess'] == True:

            df = st.session_state['input_df']
            st.session_state['dataframe'] = df
            predefined_dataset = pd.read_excel('usa_address.xlsx')

            city_state_sample, city_state_predefined = initialize(
                df, predefined_dataset)
            st.session_state['corrected_dataframe'] = preprocess(
                df, predefined_dataset, city_state_predefined)

            st.write("## Corrected States & Cities: ")

            gd = GridOptionsBuilder.from_dataframe(
                st.session_state['corrected_dataframe'])
            gd.configure_default_column(editable=True, groupable=True)
            gd.configure_selection(
                selection_mode="multiple", use_checkbox=True)
            gridoptions = gd.build()
            response = AgGrid(st.session_state['corrected_dataframe'],
                              gridOptions=gridoptions,
                              editable=True,
                              theme='balham',
                              update_mode=GridUpdateMode.MANUAL,
                              allow_unsafe_jscode=True,
                              height=200,
                              fit_columns_on_grid_load=True)

            st.markdown("""                 *Note:* 
                
                - Don't forget to hit enter ↩ on to update.
                - If you want the original value to be retained ✓ the check box. """)

            sel_rows = response['selected_rows']
            st.session_state['response'] = response


else:

    st.write(""" # Provider Data Quality Issue:
    Please upload the required dataset. """)

    uploaded_file = st.sidebar.file_uploader(
        "Choose a Excel or CSV file", type=["csv", "xslx"])

    if uploaded_file is not None:

        if 'input_df' not in st.session_state:

            dataset1 = pd.read_csv(uploaded_file)
            st.session_state['input_df'] = dataset1
            st.write("File Preview:")
            st.dataframe(dataset1, width=650, height=550)

        if st.sidebar.button(' Preprocess ↻ '):

            if 'preprocess' not in st.session_state:
                st.session_state['preprocess'] = True
                st.session_state['complete_file'] = True
                st.session_state['final'] = True

    if 'final' in st.session_state:

        if st.sidebar.button('Generate Preprocessesd File.'):
            if st.session_state['complete_file'] == True:
                data_frame = update_provider_names(
                    st.session_state['response'], st.session_state['dataframe'])

            st.session_state['preprocess'] = False
            st.markdown("""
                [Download CSV file](st.session_state['dataframe'])
            """)
            st.write('#### Complete Preprocessed File: ')
            st.dataframe(data_frame, width=650, height=550)

    if 'preprocess' in st.session_state:

        if st.session_state['preprocess'] == True:

            dict = enchant.Dict("en_US")
            dataset1 = pd.DataFrame(
                dataset1["Provider Name"], columns=["Provider Name"])
            # tokenization of provider name
            dataset1["tokenized Name"] = [
                list(word_tokenize(x)) for x in dataset1["Provider Name"]]
            # correction  string
            dataset1["Filtered Name"] = [checkAutoCorrect(
                x) for x in dataset1["tokenized Name"]]

            st.write("## Corrected Provider Names: ")
            st.session_state['dataframe'] = dataset1
            gd = GridOptionsBuilder.from_dataframe(dataset1)
            gd.configure_default_column(editable=True, groupable=True)
            gd.configure_selection(
                selection_mode="multiple", use_checkbox=True)
            gridoptions = gd.build()
            response = AgGrid(dataset1,
                              gridOptions=gridoptions,
                              editable=True,
                              theme='balham',
                              update_mode=GridUpdateMode.MANUAL,
                              allow_unsafe_jscode=True,
                              height=200,
                              fit_columns_on_grid_load=True)

            st.markdown("""                 *Note:* 
                
                - Don't forget to hit enter ↩ on to update.
                - If you want the original value to be retained ✓ the check box. """)

            sel_rows = response['selected_rows']
            st.session_state['response'] = response
