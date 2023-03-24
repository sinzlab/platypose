FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install -r requirements.txt
