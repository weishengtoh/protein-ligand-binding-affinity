import streamlit as st

st.set_page_config(page_title="Installation", page_icon='🛠️')

st.markdown("""
    # Installation 🛠️  
    Clone the repository to your local environment using 

    ```shell
    git clone https://github.com/weishengtoh/protein-ligand-binding-affinity.git
    ```
    
    ### Installing the dependencies with Conda/Docker  
    The dependencies can either be installed directly into a conda environment 
    from the `environment.yml` file, **OR** defined within a Docker image using 
    the `Dockerfile`  

""")

st.info("""
**NOTE:** Running multi-GPUs training *(single node)* requires that the dependencies 
are defined using the docker image ***(option 2 below)***  
""")

st.markdown("""

    1. **Setting up dependencies within a conda environment**  
        
        Create the conda environment from `environment.yml` file 
        ```shell
        conda env create -f environment.yml
        ```
        Activate the conda environment
        ```shell
        conda activate capstone_pytorch
        ```
    2. **Setting up dependencies within a Docker image**  

        The docker image required is located in DockerHub at  
        ```html
        https://hub.docker.com/r/weishengtoh/capstone_pytorch/tags
        ```

        To download the image into your local machine, run the following command 
        in your terminal:  
        ```shell
        docker pull weishengtoh/capstone_pytorch:horovod_v1
        ```  
        Run the docker container interatively, mounting the volume and selecting the GPU(s).  
        Make sure that the volume mounted is the path to the project root folder.

        **Single GPU:**  
        ```shell
        docker run -it -v <volume_location>:/workspace --gpus '"device=<gpu>:<mig>"' <repository_name>:<tag>
        ```

        **Multi-GPUs, Single Node:**  
        ```shell
        docker run -it -v <volume_location>:/workspace --gpus '"device=<gpu1>,<gpu2>"' <repository_name>:<tag>
        ```

    ### Weights and Biases (wandb)  
    The project uses [Weights and Biases](https://wandb.ai/site) heavily for experiment tracking, visualisations
    and model versioning. Although it is not required, it is still recommended 
    to run the training experiments with wandb support. To use wandb, you will 
    need to have an account created, and the api key ready.  

    As an alternative, TensorBoard has also been integrated into the codes, which 
    allows the experiments to be visualised locally.    
""")
