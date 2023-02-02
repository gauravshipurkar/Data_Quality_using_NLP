import streamlit as st
import pandas as pd
import enchant
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from preprocessing import filedownload, update_values, preprocess_abbreviation, preprocess, initialize, checkAutoCorrect, update_provider_names


# MAIN FUNCTION:
# ----------------------------------------------------

st.set_page_config(layout="wide")

with st.sidebar:
    st.markdown("""
    __Select a Data Attribute from list below__
    """)
    page = st.selectbox('Select:', ['Address', 'Provider'])

if page == 'Address':

    st.write(""" # Address Data Quality:
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

        if st.button('Generate Complete File..'):

            if st.session_state['abbrev'] == True:
                data_frame = update_values(
                    st.session_state['response'], st.session_state['dataframe'])
                st.session_state['dataframe'] = data_frame
                tokens = preprocess_abbreviation(data_frame)
                st.session_state['dataframe']['address'] = tokens
                st.session_state['abbrev'] = False

            st.session_state['preprocess'] = False
            # st.write(st.session_state['dataframe'])
            st.markdown(filedownload(
                st.session_state['dataframe'], decide=1), unsafe_allow_html=True)
            st.write('#### Complete Preprocessed File: ')
            st.dataframe(st.session_state['dataframe'], width=650, height=550)

    if 'preprocess' in st.session_state:

        if st.session_state['preprocess'] == True:

            df = st.session_state['input_df']
            st.session_state['dataframe'] = df
            predefined_dataset = pd.read_csv('Dataset\cities_states.csv')

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

    st.write(""" # Provider Data Quality:
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

        if st.button('Generate Preprocessesd File.'):
            if st.session_state['complete_file'] == True:

                data_frame = update_provider_names(
                    st.session_state['response'], st.session_state['dataframe'])

            st.session_state['preprocess'] = False
            st.markdown(filedownload(
                data_frame['Provider_Name'], decide=0), unsafe_allow_html=True)
            st.write('#### Complete Preprocessed File: ')
            st.dataframe(data_frame['Provider_Name'], width=650, height=550)

    if 'preprocess' in st.session_state:

        if st.session_state['preprocess'] == True:

            dict = enchant.Dict("en_US")
            dataset1 = pd.DataFrame(st.session_state['input_df']
                                    ["Provider_Name"], columns=["Provider_Name"])
            # tokenization of provider name
            dataset1["tokenized_Name"] = [
                list(word_tokenize(x)) for x in dataset1["Provider_Name"]]
            # correction  string
            dataset1["Filtered_Name"] = [checkAutoCorrect(
                x) for x in dataset1["tokenized_Name"]]

            st.write("## Corrected Provider Names: ")
            st.session_state['dataframe'] = dataset1
            gd = GridOptionsBuilder.from_dataframe(
                dataset1[["Provider_Name", "Filtered_Name"]])
            gd.configure_default_column(editable=True, groupable=True)
            gd.configure_selection(
                selection_mode="multiple", use_checkbox=True)
            gridoptions = gd.build()
            response = AgGrid(dataset1[["Provider_Name", "Filtered_Name"]],
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
