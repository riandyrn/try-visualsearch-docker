FROM awsdeeplearningteam/mxnet-model-server:latest
USER root
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/nmslib/hnsw \
    && cd hnsw  \
    && pip install pybind11 numpy setuptools \
    && cd python_bindings \
    && python setup.py install

# Because of timeouts and issues in fargate, make a BIG image with everything included
COPY index.idx /data/visualsearch/mms/index.idx
COPY idx_ASIN.pkl /data/visualsearch/mms/idx_ASIN.pkl
COPY ASIN_data.pkl /data/visualsearch/mms/ASIN_data.pkl

COPY visualsearch.mar .

CMD ["mxnet-model-server", "--start", "--models", "visualsearch=./visualsearch.mar", "--model-store", "."]