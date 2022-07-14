import streamlit as st

st.set_page_config(page_title="Installation", page_icon='🌟')

st.markdown("""
    # Usage 🌟  
    There are several entry scripts to run the training/inference. In general 
    however, the command to execute a script in the terminal has the format:   

    ```shell
    python <entry_script> -C <path_to_config_file>
    ```  

    To use horovod to train on multiple GPUs on a single node, use the command:  
    ```shell
    horovodrun -np <number_of_gpus> python <entry_script> -C <path_to_config_file>
    ```  
""")

st.markdown("""
    ### Entry Scripts

    The available entry scripts are:
    | SCRIPT | DESCRIPTION |  
    | ------ | ----------- |  
    | train.py | Runs the training on a single GPU. Training logs and models are stored locally. |
    | train_wandb.py | Runs the training on a single GPU. Training logs are uploaded to Weights and Biases (wandb) in real-time for experiment tracking and visualisation. Trained model are also uploaded to wandb for model versioning. | 
    | predict.py | Generates the prediction on a single GPU/CPU. Prediction outputs are saved in the `outputs` folder. |
    | train_horovod.py | Runs the training on multiple GPUs in a single node. Training logs are uploaded to wandb in real-time for experiment tracking and visualisation. Trained model are also uploaded to wandb for model versioning. | 

    #####  
""")

st.markdown("""
    ### Predefined Configuration Files

    Configuration files are defined in the form of a yaml file, and they are where 
    the user may choose to modify the *model, training and inference* parameters.  

    There are several predefined configurations file included in the folder `configs` that corresponds to a pretrained model in the `pretrained` folder:   
    | SMILES ENCODER | FASTA ENCODER | REGRESSOR | CONFIG FILE| TRAINED MODEL | 
    | -------------- | ------------- | --------- | ---------- | ------------- | 
    | Convolutional 1D | Convolutional 1D | Fully Connected | configs/cnn_cnn.yaml | pretrained/cnn_cnn.yaml | 
    | Graph Convolutional Network | Convolutional 1D | Fully Connected | configs/gcn_cnn.yaml | pretrained/gcn_cnn.yaml | 
    | Graph Isomorphism Network *(Attribute Masking)* | Convolutional 1D | Fully Connected | configs/ginmask_cnn.yaml | pretrained/ginmask_cnn | 
    | Graph Isomorphism Network *(Edge Existence Prediction)* | Convolutional 1D | Fully Connected | configs/ginedgepred_cnn.yaml | pretrained/ginedgepred_cnn | 
    | Graph Isomorphism Network *(Maximising Node and Pooled Global Graph Info)* | Convolutional 1D | Fully Connected | configs/gininfomax_cnn.yaml | pretrained/gininfomax_cnn |    
    
    #####  
""")
st.warning("""
    **NOTE:** The configuration files above defines the model hyperparameters that 
    are used in the pretrained models included in the folder `pretrained`. As 
    such, the ***model hyperparameters*** section **SHOULD NOT** be modified if 
    you are running the pretrained models for inference. 
""")

st.markdown("""
    Although there are only 5 pretrained models available for use, there are several other config files provided in the `configs` folder which may be used for loading and training model architectures we did not manage to fully train on.
""")

st.markdown("""
    ### Modifying / Creating Configuration Files
    Other than using the config files that were included, it is also possible to define your own config file to build and train a custom model architecture.  

    Configuration files are composed of three components:
    1. Model Hyperparameters  
    2. Training Hyperparameters  
    3. Inference Hyperparameters  

    The **model hyperparameters** section **MUST** be defined in all configuration 
    files, as this is where the model architecture is defined.  

    The **training hyperparameters** and **inference hyperparamters** sections are 
    **OPTIONAL**, depending on whether you are running training or inference.  
""")

with st.expander("Click to view an example of a full config file!"):
    st.markdown("""
        ```yaml
        # --- DEFINE THE MODEL HYPERPARAMETERS HERE --- #
        architecture: 'cnn_cnn'
        model:                                            
            smiles_config:                                  # Define the encoder architecture for smiles
                name: 'Conv1DEncoder'
                feature_dim: 100
                batch_norm: True
                out_channels: 
                - 32
                - 64
                - 96
                kernel_size:
                - 4
                - 6
                - 8
                embed_dim: 128
                activation: 'ReLU'
                dropout: 0
            sequence_config:                                # Define the encoder architecture for sequences
                name: 'Conv1DEncoder'
                feature_dim: 1000
                batch_norm: True
                out_channels: 
                - 32
                - 64
                - 96
                kernel_size:
                - 4
                - 8
                - 12
                embed_dim: 128
                activation: 'ReLU'
                dropout: 0
            regressor_config:                               # Define the regressor architecture
                name: 'DenseRegressor'
                feature_dim: 192
                out_features: 
                - 1024
                - 1024
                - 512
                - 1
                activation: 'ReLU'
                dropout: 0.1

        ### --- DEFINE THE TRAINING HYPERPARAMETERS HERE --- ###
        train:
            model_path: 'ck_pts'                            # folder to save the trained model
            data_path: 'data/RP_PLB_100000_42.parquet'      # data path
            epochs: 200                                     # number of epochs to train on
            learning_rate: 0.0003                           # initial learning rate
            seed: 42                                        # random seed
            batch_size: 32                                  # batch size for data
            val_ratio: 0.1                                  # validation ratio
            test_ratio: 0.1                                 # test ratio
            shuffle: True                                   # shuffle the data
            drop_last: True                                 # drop the last non-filled batch of data 
            label: 'Label'                                  # Target column name (binding affinity)
            smiles: 'SMILES'                                # SMILES column name (ligand/compound) 
            sequence: 'Target Sequence'                     # FASTA column name (protein sequence)

        ### --- DEFINE THE INFERENCE HYPERPARAMETERS HERE --- ###
        predict:
            ck_pt: 'pretrained/cnn_cnn'                     # path to load the trained model
            data_path: 'data/RP_PLB_100000_42.parquet'      # data path
            seed: 42                                        # random seed
            batch_size: 32                                  # batch size for data
            label: 'Label'                                  # Target column name (binding affinity) - OPTIONAL
            smiles: 'SMILES'                                # SMILES column name (ligand/compound) 
            sequence: 'Target Sequence'                     # FASTA column name (protein sequence)
        ```
    """)

st.markdown("""
    #### Model Hyperparameters
    | HEADER | DESCRIPTION |  
    | ----------- | ----------- |
    | architecture | This is where the overall architecture of the model is defined. Available options are:  `'gcn_cnn'`, `'cnn_cnn'`, `'cnn_transformers'`, `'gat_cnn'`, `'gat_transformers'`, `'gcn_transformers'`, `'gcn_esm1b'`, `'ginmask_cnn'`|
    | model.smiles_config | This is where the encoder component for the SMILES is defined. |
    | model.sequence_config | This is where the encoder component for the FASTA is defined. |  

    ###
""")

st.info("""
    More information about the signatures for the model components are included in the file `model_components.py`
""")

with st.expander("Click to view an example!"):
    st.markdown("""
        ```yaml
        # --- DEFINE THE MODEL HYPERPARAMETERS HERE --- #
        architecture: 'cnn_cnn'
        model:                                            
            smiles_config:                                  # Define the encoder architecture for smiles
                name: 'Conv1DEncoder'
                feature_dim: 100
                batch_norm: True
                out_channels: 
                - 32
                - 64
                - 96
                kernel_size:
                - 4
                - 6
                - 8
                embed_dim: 128
                activation: 'ReLU'
                dropout: 0
            sequence_config:                                # Define the encoder architecture for sequences
                name: 'Conv1DEncoder'
                feature_dim: 1000
                batch_norm: True
                out_channels: 
                - 32
                - 64
                - 96
                kernel_size:
                - 4
                - 8
                - 12
                embed_dim: 128
                activation: 'ReLU'
                dropout: 0
            regressor_config:                               # Define the regressor architecture
                name: 'DenseRegressor'
                feature_dim: 192
                out_features: 
                - 1024
                - 1024
                - 512
                - 1
                activation: 'ReLU'
                dropout: 0.1
        ```
    """)

st.markdown("""
    #### Training Hyperparameters
    | HEADER | DESCRIPTION |  
    | ------ | ----------- |
    | train.model_path | *(str)* Folder to store the trained model |
    | train.data_path | *(str)* Path to the raw data in `.parquet` |
    | train.epochs | *(int)* Max number of epochs to train on |
    | train.seed | *(int)* Seed value for reproducibility |
    | train.batch_size | *(int)* Batch size to use | 
    | train.val_ratio | *(float)* Ratio of the dataset to be used for validation. Range: [0, 1] |
    | train.test_ratio | *(float)* Ratio of the dataset to be used for testing. Range: [0, 1] |
    | train.shuffle | *(bool)* Whether to shuffle the dataset or not. Ignored if running on Horovod | 
    | train.drop_last | *(bool)* Whether to drop the last incomplete batch or not |
    | train.label | *(str)* Name of the column for the target variable (binding affinity) |
    | train.smiles | *(str)* Name of the column for the SMILES input (ligand/compound) |
    | train.sequence | *(str)* Name of the column for the FASTA input (protein sequence) |

    #####  
""")
with st.expander("Click to view an example!"):
    st.markdown("""
        ```yaml
            ### --- DEFINE THE TRAINING HYPERPARAMETERS HERE --- ###
            train:
                model_path: 'ck_pts'
                data_path: 'data/RP_PLB_100000_42.parquet'    
                epochs: 200                       
                learning_rate: 0.0003                          
                seed: 42                                       
                batch_size: 32                               
                val_ratio: 0.1                                
                test_ratio: 0.1                              
                shuffle: True                                   
                drop_last: True                               
                label: 'Label'                                
                smiles: 'SMILES'                               
                sequence: 'Target Sequence'                    
        ```
    """)

st.markdown("""
    #### Inference Hyperparameters
    | HEADER | DESCRIPTION |  
    | ------ | ----------- |
    | predict.ck_pt | *(str)* Path to the trained model to load |
    | predict.data_path | *(str)* Path to the raw data |
    | predict.seed | *(int)* Seed value for reproducibility | 
    | predict.batch_size | *(int)* Batch size to use |  
    | predict.label | *(str)* Name of the column for the target variable (binding affinity) |
    | predict.smiles | *(str)* Name of the column for the SMILES input (ligand/compound) |
    | predict.sequence | *(str)* Name of the column for the FASTA input (protein sequence) |
    
    #####  
""")
with st.expander("Click to view an example!"):
    st.markdown("""
        ```yaml
        ### --- DEFINE THE Inference HYPERPARAMETERS HERE --- ###
        predict:
            ck_pt: 'pretrained/cnn_cnn'           
            data_path: 'data/RP_PLB_100000_42.parquet'    
            seed: 42                            
            batch_size: 32                       
            label: 'Label'             
            smiles: 'SMILES'                             
            sequence: 'Target Sequence'                   
        ```
    """)