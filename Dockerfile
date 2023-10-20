FROM sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9

WORKDIR /src

RUN pip install --no-cache-dir --upgrade keyrings.alt
RUN pip install --no-cache-dir poetry


# ADD .git to image to allow for commit hash retrieval
ADD . /src

RUN poetry install
RUN poetry run python -m pip install chumpy
