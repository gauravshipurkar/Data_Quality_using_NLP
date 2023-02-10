import streamlit as st
import pandas as pd
import enchant
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from preprocessing import Preprocess
from clustering import Clustering

def model_declare():

    model_path = 'local/path/to/model'
    model = SentenceTransformer(model_path)

    return model

# MAIN FUNCTION:
# ----------------------------------------------------

def __main__():

    try:
        st.set_page_config(layout="wide")
        
        with st.sidebar:
            st.markdown(""" __Select a Data Attribute from list below__ """)
            page = st.selectbox('Select:', ['Address', 'Provider'])

        if page == 'Address':
            
            """
                Choosing Address in the drop down will execute the further code
            """

            st.write(""" # Address Data Quality:
                Please upload the required dataset.""")

            uploaded_file = st.sidebar.file_uploader("Choose a Excel or CSV file", type=["csv", "xlsx"])

            if uploaded_file is not None:

                if 'input_dataframe' not in st.session_state:

                    st.write(""" ##### File Preview:""")
                    st.session_state['input_dataframe'] = pd.read_csv(uploaded_file)
                    st.dataframe(st.session_state['input_dataframe'], width=650, height=550)

                if st.sidebar.button(' Preprocess ↻ '):

                    if 'preprocess' not in st.session_state:

                        st.session_state['preprocess'] = True
                        st.session_state['abbreviation'] = True
                        st.session_state['show_abbreviation'] = True

            if 'show_abbreviation' in st.session_state:

                if st.sidebar.button('Generate Complete File..'):

                    """
                        This button will correct all the common abbreviations present in the file such as "Rd, Hwy"
                    """

                    if st.session_state['abbreviation'] == True:
                        
                        data_frame = Preprocess.update_values(st.session_state['response'], st.session_state['dataframe'])
                        tokens = Preprocess.preprocess_abbreviation(data_frame)
                        st.session_state['dataframe']['address'] = tokens
                        st.session_state['dataframe'] = data_frame
                        st.session_state['abbreviation'] = False
                        st.session_state['show_cluster'] = True
                        st.session_state['model'] = model_declare()

                    st.session_state['preprocess'] = False
                    st.markdown(Preprocess.filedownload(st.session_state['dataframe'], decide=1), unsafe_allow_html=True)
                    st.write('#### Complete Preprocessed File: ')
                    st.dataframe(st.session_state['dataframe'], width=650, height=550)

            if 'show_cluster' in st.session_state:
                
                if st.sidebar.button('Proceed for Clustering'):
                    """
                        This is the code for the clustering algorithm, which checks the similarity score, further
                        processes to checking of the numbers in the code and if similar, groups them together and forms
                        a cluster.
                    """    
                    st.session_state['Cluster_Assign'] = [0 for i in range(0,len())]
                    Cluster = {}
                    threshold = 0.75

                    dataframe = st.session_state['dataframe']
                    Address = Clustering.convert_to_single(dataframe['address'], dataframe['city'], dataframe['state'], dataframe['zip'])

                    for main_address in range(0, len(Address)):
                        print(Address[main_address])
                        for loop_address in range(0, len(Address)):

                            if Address[main_address] == Address[loop_address]:

                                continue

                            else:

                                main_string, second_string = Clustering.Digit_Removal(Address[main_address], Address[loop_address])
                                main_embeddings = st.session_state['model'].encode([main_string])[0]
                                second_embeddings = st.session_state['model'].encode([second_string])[0]
                                similarity_percentage = Clustering.cosine_function(main_embeddings, second_embeddings)

                                if similarity_percentage >= threshold:

                                    check = Clustering.Digit_Comparision(Address[main_address], Address[loop_address])
                                    
                                    if check:

                                        Cluster, length, st.session_state['Cluster_Assign'] = Cluster_Formation( Address[main_address], Address[loop_address], Cluster, length, st.session_state['Cluster_Assign'], main_address, loop_address)

                    dataframe['Clusters'] = st.session_state['Cluster_Assign']
                    st.markdown(filedownload(dataframe, decide=1),unsafe_allow_html=True)
                    st.dataframe(dataframe, width=650, height=550)

            if 'preprocess' in st.session_state:

                if st.session_state['preprocess'] == True:

                    """
                        This is the code for spell-correction and pre-processing which will correct the values of 
                        the data-attribute respectively.
                    """

                    df = st.session_state['input_dataframe']
                    st.session_state['dataframe'] = df
                    predefined_dataset = pd.read_csv('Dataset\cities_states.csv')

                    city_state_sample, city_state_predefined = Preprocess.initialize(df, predefined_dataset)
                    st.session_state['corrected_dataframe'] = Preprocess.correct_change(df, predefined_dataset, city_state_predefined)
                    
                    st.write("## Corrected States & Cities: ")

                    """
                        This will show the grid-table having the original values and also the correct values
                        which are completely editable.
                    """

                    gd = GridOptionsBuilder.from_dataframe(st.session_state['corrected_dataframe'])
                    gd.configure_default_column(editable=True, groupable=True)
                    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
                    gridoptions = gd.build()
                    response = AgGrid(st.session_state['corrected_dataframe'], gridOptions=gridoptions, editable=True, theme='balham', update_mode=GridUpdateMode.MANUAL, allow_unsafe_jscode=True, height=200, fit_columns_on_grid_load=True)

                    st.markdown("""                 *Note:* 
                        
                        - Don't forget to hit enter ↩ on to update.
                        - If you want the original value to be retained ✓ the check box. """)

                    st.session_state['response'] = response


        else:

            st.write(""" # Provider Data Quality:
                    Please upload the required dataset.""")

            uploaded_file = st.sidebar.file_uploader(
                "Choose a Excel or CSV file", type=["csv", "xslx"])

            # if uploaded_file is not None:

            #     if 'input_df' not in st.session_state:

            #         dataset1 = pd.read_csv(uploaded_file)
            #         st.session_state['input_df'] = dataset1
            #         st.write("File Preview:")
            #         st.dataframe(dataset1, width=650, height=550)

            #     if st.sidebar.button(' Preprocess ↻ '):

            #         if 'preprocess' not in st.session_state:
            #             st.session_state['preprocess'] = True
            #             st.session_state['complete_file'] = True
            #             st.session_state['final'] = True

            # if 'final' in st.session_state:

            #     if st.button('Generate Preprocessesd File.'):
            #         if st.session_state['complete_file'] == True:

            #             data_frame = update_provider_names(
            #                 st.session_state['response'], st.session_state['dataframe'])

            #         st.session_state['preprocess'] = False
            #         st.markdown(filedownload(
            #             data_frame['Provider_Name'], decide=0), unsafe_allow_html=True)
            #         st.write('#### Complete Preprocessed File: ')
            #         st.dataframe(data_frame['Provider_Name'], width=650, height=550)

            # if 'preprocess' in st.session_state:

            #     if st.session_state['preprocess'] == True:

            #         dict = enchant.Dict("en_US")
            #         dataset1 = pd.DataFrame(st.session_state['input_df']
            #                                 ["Provider_Name"], columns=["Provider_Name"])
            #         # tokenization of provider name
            #         dataset1["tokenized_Name"] = [
            #             list(word_tokenize(x)) for x in dataset1["Provider_Name"]]
            #         # correction  string
            #         dataset1["Filtered_Name"] = [checkAutoCorrect(
            #             x) for x in dataset1["tokenized_Name"]]

            #         st.write("## Corrected Provider Names: ")
            #         st.session_state['dataframe'] = dataset1
            #         gd = GridOptionsBuilder.from_dataframe(
            #             dataset1[["Provider_Name", "Filtered_Name"]])
            #         gd.configure_default_column(editable=True, groupable=True)
            #         gd.configure_selection(
            #             selection_mode="multiple", use_checkbox=True)
            #         gridoptions = gd.build()
            #         response = AgGrid(dataset1[["Provider_Name", "Filtered_Name"]],
            #                           gridOptions=gridoptions,
            #                           editable=True,
            #                           theme='balham',
            #                           update_mode=GridUpdateMode.MANUAL,
            #                           allow_unsafe_jscode=True,
            #                           height=200,
            #                           fit_columns_on_grid_load=True)

            #         st.markdown("""                 *Note:* 
                        
            #             - Don't forget to hit enter ↩ on to update.
            #             - If you want the original value to be retained ✓ the check box. """)

            #         sel_rows = response['selected_rows']
            #         st.session_state['response'] = response

    except Exception as err:
        print(err)

__main__()