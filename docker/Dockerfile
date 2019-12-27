FROM continuumio/anaconda3
MAINTAINER "Andrei Maksimov"

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    /opt/conda/bin/conda install jupyter -y 

RUN /opt/conda/bin/conda install numpy pandas scikit-learn -y

RUN /opt/conda/bin/conda install matplotlib seaborn -y

RUN pip install opencv-python 

RUN /opt/conda/bin/conda install pyyaml h5py keras -y && \
    /opt/conda/bin/conda upgrade dask && \
    pip install tensorflow imutils

RUN ["mkdir", "notebooks"]

COPY conf_jupyter /root/.jupyter
COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
VOLUME /notebooks

CMD ["/run_jupyter.sh"]
