FROM sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install -r requirements.txt
RUN pip install torch-geometric
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip install lipstick
RUN pip install wandb