FROM tensorflow/tensorflow
WORKDIR /root

ARG GITHUB_OWNER_ARG
ENV GITHUB_OWNER=${GITHUB_OWNER_ARG}

ARG GITHUB_REPO_ARG
ENV GITHUB_REPO=${GITHUB_REPO_ARG}

ARG GITHUB_TOKEN_ARG
ENV GITHUB_TOKEN=${GITHUB_TOKEN_ARG}

# Copies the trainer code
RUN mkdir /root/trainer
COPY train.py /root/trainer/mnist.py

RUN pip install PyGithub

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "trainer/mnist.py"]