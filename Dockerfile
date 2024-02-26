FROM sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install --no-cache-dir --upgrade keyrings.alt
RUN pip install --no-cache-dir poetry

RUN poetry config installer.max-workers 10

RUN poetry install --no-interaction --no-ansi -vvv
RUN poetry add pandas