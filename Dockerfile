FROM tensorflow/tensorflow
WORKDIR /root

# Copies the trainer code
RUN mkdir /root/trainer
COPY train.py /root/trainer/mnist.py

ENV GOOGLE_APPLICATION_CREDENTIALS $GOOGLE_APPLICATION_CREDENTIALS

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "trainer/mnist.py"]