from mxnet_model_service import MXNetModelService
from mxnet.gluon.data import ArrayDataset
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import base64
from PIL import Image
import io
import json
import pickle
import logging
import os

# External dependencies
import hnswlib

# Load the network
ctx = mx.cpu()

# Fixed parameters
SIZE = (224, 224)
MEAN_IMAGE = mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])
EMBEDDING_SIZE = 512
EF = 300
K = 25

# Data Transform


def transform(image):
    resized = mx.image.resize_short(image, SIZE[0]).astype('float32')
    cropped, crop_info = mx.image.center_crop(resized, SIZE)
    cropped /= 255.
    normalized = mx.image.color_normalize(cropped,
                                          mean=MEAN_IMAGE,
                                          std=STD_IMAGE)
    transposed = nd.transpose(normalized, (2, 0, 1))
    return transposed


class VisualSearchService(MXNetModelService):

    def initialize(self, context):
        super(VisualSearchService, self).initialize(context)
        data_dir = os.environ.get('DATA_DIR', '/data/visualsearch/mms/')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        ############################################
        logging.info('Loading Resources files')

        self.idx_ASIN = pickle.load(
            open(os.path.join(data_dir, 'idx_ASIN.pkl'), 'rb'))
        self.ASIN_data = pickle.load(
            open(os.path.join(data_dir, 'ASIN_data.pkl'), 'rb'))
        self.p = hnswlib.Index(space='l2', dim=EMBEDDING_SIZE)
        self.p.load_index(os.path.join(data_dir, 'index.idx'))
        ############################################

        logging.info('Resources files loaded')

        self.p.set_ef(EF)
        self.k = K

    def preprocess(self, data):
        for key in data[0]:
            logging.info(
                f'[RDebug] key: {key}, type(data[0][{key}]): {type(data[0][key])}')

        image_bytes = data[0]['body']
        image_PIL = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image_PIL)
        image_t = transform(nd.array(image_np[:, :, :3]))
        image_batchified = image_t.expand_dims(axis=0).as_in_context(ctx)
        logging.info(f'[RDebug] image_batchified: {image_batchified}')
        return [image_batchified]

    def postprocess(self, data):
        labels, distances = self.p.knn_query(
            [data[0].asnumpy().reshape(-1,)], k=self.k)
        logging.info(labels)
        output = []
        for label in labels[0]:
            ASIN = self.idx_ASIN[label]
            output.append(self.ASIN_data[ASIN])
        logging.info(f"[RDebug] output: {output}")
        return [output]


_service = VisualSearchService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
