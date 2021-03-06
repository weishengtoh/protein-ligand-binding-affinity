import os
import sys

sys.path.append('.')
import pandas as pd
import streamlit as st
import yaml

from model.custom_model import CustomModel, Models

st.set_page_config(page_title="Single Prediction", page_icon="📁")

st.write("# Single Prediction")
st.markdown("""
    To make predictions on a single protein/ligand pair, input the compound in 
    SMILES encoding, and the protein FASTA sequence below. 👇

""")

with st.sidebar:
    st.write("## Model Selection")
    selected_model = st.radio(
        'Choose your model',
        [
            'cnn_cnn', 'gcn_cnn', 'ginmask_cnn', 'ginedgepred_cnn',
            'gininfomax_cnn'
        ],
        help="""Currently these are the only trained models available. 
        \nHowever, you may also build your own models using the source file.""",
    )

config_path = os.path.join('configs', f'{selected_model}.yaml')

with st.form(key='my_form'):
    smiles = st.text_input('SMILES', help='The compound in SMILES format.')
    fasta = st.text_input('FASTA', help='The protein in FASTA format.')
    predict = st.form_submit_button('Predict!')

if predict:
    ### --- LOAD THE CONFIG FILE --- ###
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config['predict']['batch_size'] = 1

    ### --- SELECT MODEL ARCHITECTURE --- ###
    architecture = Models(selected_model)

    ### --- GET MODEL --- ###
    model = CustomModel(architecture=architecture, config=config)

    ### --- LOAD DATA --- ###
    SMILES = 'smiles'
    SEQUENCE = 'sequence'
    df = pd.DataFrame(data={SMILES: [smiles], SEQUENCE: [fasta]})
    dataloader_samplers = model.load_data(dataframe=df,
                                          smiles=SMILES,
                                          sequence=SEQUENCE,
                                          label=None,
                                          mode='predict')
    dataloader, _, _ = dataloader_samplers['dataloaders']

    ### --- LOAD MODEL FROM CK_PT -- ##
    model.load_ckpt(config['predict']['ck_pt'])

    ### --- GENERATE PREDICTIONS --- ###
    with st.spinner(text='Generating Predictions...'):
        preds = model.predict(dataloader)

    st.success("""
        Congrats! 🥳 The predictions has been generated. 
    """)

    st.metric('Predicted Binding Affinity', f'{preds:.4}')
