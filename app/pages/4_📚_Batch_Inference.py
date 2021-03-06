import os
import random
import sys
from typing import Any, Dict

sys.path.append('.')

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchmetrics
import yaml
from metrics.metrics import EnrichmentFactor, KendallTau
from model.custom_model import CustomModel, Models
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from torch import nn
from utils.history import History

st.set_page_config(page_title="Batch Inference", page_icon="📚")


def set_seed(seed):
    """Set the seed for reproducibility"""

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def obtain_metrics() -> Dict[str, Any]:

    metrics = {
        'rmse': torchmetrics.MeanSquaredError(squared=False),
        'pearson': torchmetrics.PearsonCorrCoef(),
        'r2score': torchmetrics.R2Score(),
        'mae': torchmetrics.MeanAbsoluteError(),
        'ef_1': EnrichmentFactor(1),
        'ef_5': EnrichmentFactor(5),
        'ef_10': EnrichmentFactor(10),
        'k_tau': KendallTau()
    }
    return metrics


@st.cache
def load_data(data_path, file_type: str):

    if file_type == 'csv':
        dataframe = pd.read_csv(data_path)
    elif file_type == 'parquet':
        dataframe = pd.read_parquet(data_path)

    return dataframe


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


with st.sidebar:

    st.write("## Upload Data")

    label_present = st.checkbox(
        'Label Present',
        help='Indicates if the raw data contains the ground truth label')

    smiles = st.text_input('Smiles:',
                           placeholder='None',
                           help='Name of the column for the smiles input')
    sequence = st.text_input('Fasta Sequence:',
                             placeholder='None',
                             help='Name of the column for the fasta input')
    label = st.text_input(
        'Target:',
        placeholder='None',
        disabled=not label_present,
        help='Name of the column for the target input (optional)')

    file_type = st.radio(
        'File format',
        ['csv', 'parquet'],
        help="File extension type",
    )

    uploaded_file = st.file_uploader("Choose a file")

st.write("# Batch Inference")
st.markdown("""
    
    **👈 To start off, upload your data in the sidebar** to experiment with some of the pretrained models!
    
""")

if uploaded_file is not None:
    df = load_data(uploaded_file, file_type=file_type)
    st.success(
        'Congrats! 🥳 Your data has been uploaded sucessfully. You may hide the sidebar and proceed below. '
    )

if uploaded_file is not None:
    st.write("### Exploratory Data Analysis 🔎")
    st.markdown("""
        Data exploration is one of the key steps in any machine learning and data 
        science problem.   
        
        Here we can scroll through the raw data interactively to confirm that 
        it has been loaded correctly.  
    """)

    if st.checkbox('Show Raw Data'):
        with st.spinner(text='Loading data into dataframe...'):

            smiles_len = df[smiles].apply(len)
            seq_len = df[sequence].apply(len)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Number of Instances', f'{len(df):,}')
            with col2:
                st.metric('SMILES Length',
                          f'{smiles_len.min()} to {smiles_len.max()}')
            with col3:
                st.metric('FASTA Length', f'{seq_len.min()} to {seq_len.max()}')
            st.subheader('Raw data')
            st.write(df)

    profile_df = df.copy()

    profile_df['smiles_len'] = profile_df[smiles].apply(len)
    profile_df['sequence_len'] = profile_df[sequence].apply(len)

    profile = ProfileReport(profile_df,
                            title="Pandas Profiling Report",
                            explorative=True,
                            progress_bar=True)

    st.markdown("""
        To accelerate our EDA workflow, we may also make use of the 
        `pandas-profiling` tool to perform our exploratory data analysis.  

    """)

    if st.checkbox('Show Pandas Profiling Report'):
        with st.spinner(text='Generating pandas profiling report...'):
            export = profile.to_html()
            col1, col2 = st.columns([10, 3])
            with col2:
                st.download_button(label="Download Report",
                                   data=export,
                                   help='Download the report as a html file',
                                   file_name='report.html')
            st_profile_report(profile)

    st.markdown("""
        By default, the pipeline implemented will drop all instances with *missing* 
        or *inf* values in any of the columns. 
    """)
    st.warning("""
        **NOTE:** Here, we should have implemented a check to ensure that the 
        schema of the data fed into the models are as expected. However, with 
        the lack of domain knowledge on what constitutes a valid protein/ligand, 
        there is no existing check to validate the input data schema. We leave 
        it up to the user to ensure that the data schema is correct. 
    """)

if uploaded_file is not None:
    st.write("### Model Selection 📈")
    st.markdown("""
        Once we are satisfied with the initial data exploration, we may proceed 
        to select one of the available pretrained models to perform batch inference.    
    """)
    col1, col2 = st.columns([3, 1])

    with col2:
        selected_model = st.radio(
            'Choose your model', [
                'cnn_cnn', 'gcn_cnn', 'ginmask_cnn', 'ginedgepred_cnn',
                'gininfomax_cnn'
            ],
            help="""Currently these are the only trained models available. 
            However, you may also build your own models using the source code."""
        )

    assets_folder = os.path.join('app', 'assets')
    with col1:
        st.image(os.path.join(assets_folder, 'placeholder.png'),
                 caption='Image of Model Here')

if uploaded_file is not None:
    st.write("### Configuration ⚙️")

    col1, col2 = st.columns([2, 5])
    with col1:
        data_samples = st.number_input('Sample Size',
                                       min_value=1,
                                       max_value=len(df),
                                       help=f'Min: 1, Max: {len(df):,}',
                                       value=len(df))
    with col2:
        st.markdown("""
            For testing purposes, you may wish to execute inference on only a 
            smaller subsample of the dataset provided. Setting the value to max 
            will execute inference on the entire dataset.
        """)

    col1, col2 = st.columns([2, 5])
    with col1:
        seed = st.number_input('Random Seed',
                               min_value=1,
                               max_value=None,
                               value=42)
    with col2:
        st.markdown("""
            The subsample is obtained randomly from the dataset. Here, you may 
            seed the randomisation to allow the results to be **reproducible**. The 
            seed is ignored if the entire dataset is being used for inference.
        """)

    col1, col2 = st.columns([2, 5])
    with col1:
        batch_size = st.select_slider(
            'Batch Size',
            options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            help='Batch size is expressed in power of 2',
            value=2048)
    with col2:
        st.markdown("""
            Batch size can be selected here, and optimally it should be increased 
            to as large as possible (subject to memory constraint) to speed up batch inference.
        """)

    col1, col2 = st.columns([10, 1])
    with col2:
        predict = st.button('Predict!')

if uploaded_file is not None and predict:
    st.write("### Results 📝")
    st.markdown("""
        Here the predictions are being generated using the selected model and the 
        configurations specified.  

        If the ground truth label has also been provided, you will also see the 
        performance metrics generated.  
    """)

    ### --- LOAD THE CONFIG FILE --- ###
    config_path = os.path.join('configs', f'{selected_model}.yaml')
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config['predict']['batch_size'] = batch_size
    config['predict']['seed'] = seed

    ### --- SET SEED FOR REPRODUCIBILITY -- ##
    set_seed(config['predict']['seed'])

    ### --- SELECT MODEL ARCHITECTURE --- ###
    architecture = Models(selected_model)

    ### --- GET MODEL --- ###
    model = CustomModel(architecture=architecture, config=config)

    ### --- LOAD DATA --- ###
    rand_ind = np.random.permutation(len(df))[:data_samples]
    df = df.iloc[rand_ind]
    dataloader_samplers = model.load_data(
        dataframe=df,
        smiles=smiles,
        sequence=sequence,
        label=label if label_present else None,
        mode='predict')
    dataloader, _, _ = dataloader_samplers['dataloaders']

    ### --- LOAD MODEL FROM CK_PT -- ##
    model.load_ckpt(config['predict']['ck_pt'])

    ### --- GENERATE PREDICTIONS --- ###
    with st.spinner(text='Generating predictions...'):
        preds = model.predict(dataloader)

    ### --- SAVE PREDICTIONS --- ###
    df['Pred'] = preds
    if label_present and len(df) > 1:
        with st.spinner(text='Computing metrics...'):

            ### --- GENERATE METRICS (OPTIONAL) --- ###
            test_metrics = obtain_metrics()
            loss = nn.MSELoss()
            test_history = model.evaluate(dataloader, loss, test_metrics)

            history_dict = test_history.get_params()
            col_len = len(history_dict)

            ### --- SHOW RESULTS --- ###
            first_row = list(st.columns(5))
            for col in first_row:
                key, value = history_dict.popitem()
                with col:
                    st.metric(key.replace('test_', '').upper(), f'{value:.4}')

            sec_row = list(st.columns(4))
            for col in sec_row:
                key, value = history_dict.popitem()
                with col:
                    st.metric(key.replace('test_', '').upper(), f'{value:.4}')

    csv = convert_df(df)
    st.success("""
        Congrats! The predictions has been generated. 
        **Click the button below** to download the results. 👇 
    """)

    col1, col2 = st.columns([14, 3])
    with col2:
        st.download_button(
            label="Save Results!",
            help='Save the predictions as csv file',
            data=csv,
            file_name='results.csv',
            mime='text/csv',
        )
