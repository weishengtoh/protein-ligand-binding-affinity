import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Introduction", page_icon='🧬')

st.markdown("""
    # Protein Ligand Binding Affinity 🧬  

    The purpose of this project is to develop and train artificial intelligence 
    models that are able to accurately predict how well a ligand will bind to a protein.  

    Binding between a protein-ligand pair is typically measured using a scalar 
    value known as the “dissociation constant”, which will be the target label 
    in the **regression problem**.  
    
    Hence in this project, the aim is to create and/or use existing models that 
    predicts the dissociation constant for any given protein-ligand pair.  
  
""")

st.markdown("""
    ### Dataset 📦

    The dataset we were provided with consisted of **100,000** instances of protein-ligand 
    pair, along with their binding affinity.  

    Each instance in the dataset contains **two independent variables** and **one 
    target variable**.  
    
    The independent variables are the **protein sequence** and the 
    **compound sequence**, which are in the [FASTA](https://en.wikipedia.org/wiki/FASTA_format) 
    and [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) 
    format respectively.  


    | VARIABLE | FORMAT | DESCRIPTION | TYPE |
    | -------- | ------ | ----------- | ------- |
    | Ligand / Compound | SMILES | Potential Drug Compound | String | 
    | Protein | FASTA | Protein Sequence | String |
    | Affinity *(Target)* | Numerical | Binding Affinity | Float |

    #####  
""")
with st.expander("Example"):
    df = pd.DataFrame(
        np.array([[
            'N=C(O)Nc1sc(-c2cccs2)cc1C(O)=N[C@H]1CCCNC1',
            'MSYKPNLAAHMPAAALNAAGSVHSPSTSMATSSQYRQLLSDYGPPSLGYTQGTGNSQVPQSKYAELLAIIEELGKEIRPTYAGSKSAMERLKRGIIHARGLVRECLAETERNARS',
            3.748188
        ]]),
        columns=['SMILES', 'FASTA', 'AFFINITY'],
    )
    st.write(df)

st.markdown("""
    We can also gain some insights by plotting the distribution of the independent 
    and target variables.  

""")

assets_folder = os.path.join('app', 'assets')

col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(assets_folder, 'smiles.png'),
             caption='SMILES Distribution')
with col2:
    st.image(os.path.join(assets_folder, 'fasta.png'),
             caption='FASTA Distribution')

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(os.path.join(assets_folder, 'target.png'),
             caption='Target Distribution')

st.markdown("""
    To handle the SMILES and FASTA inputs with varying length, we perform 
    truncation and padding to length **100** and length **1000** respectively.  

    The truncation and padding length also follows the typical approach used in 
    most research papers covered, and in some cases is also guided by the 
    limitations imposed by other pretrained models which we have used.  

    It has also been observed that increasing the truncation and padding length 
    significantly increases the training time, while the performance of the model 
    remains relatively unchanged.   

    The distribution of the target variable that we have used in our training is 
    also important, as most of the affinity in our dataset are within 0 to 5, with 
    a noticable peak around values 3 to 4. This will have a significant impact on 
    the model generalisability, especially when used on datasets such as 
    [Kiba](http://iilab.hit.edu.cn/dtadata/ElectraDTA/dataset/kiba-full-data.csv) or 
    [Davis](http://iilab.hit.edu.cn/dtadata/ElectraDTA/dataset/davis-full-data.csv), 
    both of which have a very different distribution from the data we were provided with.  

    To work with the data, we have adopted a ***80/10/10*** segregation strategy for the 
    ***train/val/test*** splits. The data is randomly shuffled during segregation, and 
    the randomisation is seeded to allow the results to be reproducible.  
""")

st.markdown("""
    ### Approach 🔬
    This project introduces the use of modular model components to build the 
    overall model architecture.  

    This is inspired by the observation that the models used for protein-ligand 
    binding affinity predictions are usually composed of two different components:  
        1. an `encoder` component to extract the feature representation 
        for the protein/ligand, **AND**  
        2. a `regressor` component used to obtain a final regression 
        output (binding affinity)

    Instead of recreating hundreds of models to experiment with, a set of 
    **encoders** and **regressors** are defined which may be used to build the 
    overall model architecture dynamically.  

    The individual model components are themselves configurable, allowing the 
    *number of layers*, *activation function*, *dropout ratio* etc. to be determined 
    dynamically during runtime in the form of a configuration file.  

    Training hyperparameters such as *learning rate*, *batch size*, *seed*, *max no. of epochs* 
    etc. can also be modified in the config files.  

    The same goes for inference hyperparameters, which can also be modified using the 
    config files.  

    `Early stopping` and `Model Checkpoint` functionalities are also provided as callbacks, 
    which by default stops the training after 20 epochs of no improvement, and saves only 
    the best model that has been observed.  
""")

st.markdown("""
    ### Results 📝 
""")

col1 = [
    'Conv1D', 'Conv1D', 'Fully_Connected', '74', '0.6945 (worst)',
    '0.6331 (worst)', '0.7997 (worst)', '0.4358', '0.6089 (worst)',
    '36.7347 (worst)', '11.8876', '5.5912'
]

col2 = [
    'Graph_Convolutional', 'Conv1D', 'Fully_Connected', '110', '0.6242 (best)',
    '0.6971 (best)', '0.8351 (best)', '0.3997 (best)', '0.6418 (best)',
    '57.1429 (best)', '11.5663', '5.5711 (worst)'
]

col3 = [
    'DGL_GIN_AttrMasking', 'Conv1D', 'Fully_Connected', '342', '0.634',
    '0.6911', '0.8345', '0.4209', '0.6267', '54.5455', '12.5451',
    '6.1824 (best)'
]

col4 = [
    'DGL_GIN_EdgePred', 'Conv1D', 'Fully_Connected', '255', '0.643', '0.6822',
    '0.8315', '0.4308', '0.6255', '52', '12.24', '5.93'
]

col5 = [
    'DGL_GIN_InfoMax', 'Conv1D', 'Fully_Connected', '290', '0.636', '0.6886',
    '0.8337', '0.4277', '0.627', '48.4848', '12.7856 (best)', '6.0621'
]

col6 = [
    'DGL_GIN_ContextPred', 'Conv1D', 'Fully_Connected', '209', '0.6555',
    '0.6698', '0.8245', '0.4407 (worst)', '0.6164', '53.5354',
    '11.4629 (worst)', '5.7916'
]

results = pd.DataFrame()
results['cnn_cnn'] = col1
results['gcn_cnn'] = col2
results['ginmask_cnn'] = col3
results['ginedgepred_cnn'] = col4
results['gininfomax_cnn'] = col5
results['gincontextpred_cnn'] = col6

index = [
    'smiles_encoder', 'sequence_encoder', 'regressor', 'epochs', 'rmse',
    'r2score', 'pearson', 'mae', "kendall's tau", 'enrichment factor* 1%',
    'enrichment factor* 5%', 'enrichment factor* 10%'
]
results.index = index

st.dataframe(results)

st.caption(
    '**enrichment factor is a custom metric that was requested by the host company*'
)

st.markdown("""
    ### Key Tools Used 🪛 
    
    For GPU compute, we were given access to an **A100 40GB** server in MIG mode for *single 
    node, single gpu* training and access to a **V100 32GB** server for *single node, multi-gpu* 
    training.  

    | FUNCTION | TOOLS |
    | -------- | ----- |
    | Exploratory Data Analysis | `pandas` `numpy` `matplotlib` `seaborn` `pandas-profiling` |
    | Modelling Framework | `pytorch` `torchmetrics` `torch_geometric` `dgl` `dgllife` `rdkit` `transformers` |
    | Experiment Visualisation | `tensorboard` `wandb` | 
    | Configuration Files Management | `hydra*` `yaml` | 
    | Model Versioning, Experiments Tracking | `wandb` |
    | Distributed Version Control | `git` `github` | 
    | Environment Management | `conda` `docker` |
    | Web App | `streamlit` |
    | Hyperparameter Search | `optuna*` |
    | Multi-GPUs Training | `horovod` |  

    #####
""")

col1, col2 = st.columns([2, 8])
with col2:
    st.caption(
        '**hydra and optuna were used internally by the team, but not part of the final deliverable.*'
    )

st.markdown("""
    ### Limitations and Recommendations 📢

    Due to the limited time that we have for the final project deliverable, we 
    were unable to fully experiment with some of the model architectures which 
    appears to be promising.  

    However, we have included the configuration files in the folder `configs`, 
    which allows the user to build and train using the architectures that were 
    predefined. The way the code infrastructure has been built also allows the 
    user to easily tweak the model architecture by only modifying the config files. 

    For this project, we have also assumed that the data is ***"fixed"*** while 
    experimenting with different model architectures. This is due to the nature of 
    the project, as we were experimenting to see if the model is able to generalise 
    well even when trained on a limited set of data.  
    
    It is obvious when plotting the distribution of the data that the models will 
    unlikely generalise well if used on other widely available datasets (such as 
    Kiba or Davis), which has a very different distribution.  

    This can be resolved by retraining the model on the dataset required, or to 
    obtain a self-curated dataset which has a less skewed distribution for thr target 
    variable.  

    Also given the lack of domain knowledge, we were unable to introduce data validation 
    in the models. The models will therefore not recognise if a given input is a 
    valid protein/ligand, and provide an output as long as the characters used are 
    in the vocabulary.   

    Finally, the data segregation adopted by the team (80/10/10 random split) might 
    not be optimal as the final performance observed might be due to a random lucky/unlucky 
    seed. The obvious mitigation is to either:
    1) use cross-validation (which requires more time than we can afford), or 
    2) handcraft the validation and test set using domain knowledge, to ensure that it is representative of the population  

""")

st.markdown("""
    ### References 📋  

    #### **DeepDTA**
    - [GitHub](https://github.com/hkmztrk/DeepDTA)  
    - [Paper](https://arxiv.org/abs/1801.10193)  

    #### **DeepDTAF**  
    - [GitHub](https://github.com/KailiWang1/DeepDTAF)  
    - [Paper](https://academic.oup.com/bib/article-abstract/22/5/bbab072/6214647?redirectedFrom=fulltext)  

    #### **GraphDTA**  
    - [GitHub](https://github.com/thinng/GraphDTA)  
    - [Paper](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)

    #### **Evolutional Scale Modelling** 
    - [GitHub](https://github.com/facebookresearch/esm)  
    - [Paper](https://www.biorxiv.org/content/10.1101/622803v4)  

    #### **ProtTrans**
    - [GitHub](https://github.com/agemagician/ProtTrans)  
    - [Paper](https://arxiv.org/abs/2007.06225)  

""")
