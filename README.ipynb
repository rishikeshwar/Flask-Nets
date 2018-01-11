# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request
from flask_restful import Resource, Api

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


import argparse
import os.path
import re
import sys
import tarfile
import json

import numpy as np
from six.moves import urllib
import tensorflow as tf

import classify_keras as ck



FLAGS = None
JSONE = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

########################################################

app = Flask(__name__)
api = Api(app)

todos = {}

class TodoTF(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def post(self):
        J = request.form.to_dict(flat=False)
               
        if 'image_file' in J:
          image = J['image_file'][0]
        else:
          image = os.path.join(J['model_dir'][0], 'cropped_panda.jpg')
        if 'num_top_predictions' not in J:
            J.update({'num_top_predictions':[5]})
        if 'model_dir' not in J:
            J.update({'model_dir': ['/tmp/imagenet']})
        print('\n\n\n\n\n\n')
        print(J)
        rishi(J)
        maybe_download_and_extract(JSONE)
        p = run_inference_on_image(image, J)
        return p

api.add_resource(TodoTF, '/api/imagenet')

class TodoKeras(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def post(self):
        J = request.form.to_dict(flat=False)
        p = ck(J)
        return p

api.add_resource(TodoKeras, '/api/resnet')

#########################################################
def rishi(J):
  global JSONE
  JSONE = J

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          JSONE['model_dir'][0], 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          JSONE['model_dir'][0], 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    print("load")
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
    print("load")  
    return node_id_to_name


  def id_to_string(self, node_id):
    print("id_to_string")
    if node_id not in self.node_lookup:
      return ''
    print("id_to_string")
    return self.node_lookup[node_id]


def create_graph(JSONE):
  print("create_graph'")
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      JSONE['model_dir'][0], 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  print("create_graph")

def run_inference_on_image(image, JSONE):
  print("run_inference_on_image")
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph(JSONE)
  sendJson = []
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-int(JSONE['num_top_predictions'][0]):][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      sendJson.append({"String": human_string, "Score": round(float(score), 4)})
      print('%s (score = %.5f)' % (human_string, score))
  print("run_inference_on_image")
  return(sendJson)


def maybe_download_and_extract(JSONE):
  """Download and extract model tar file."""
  print("maybe_download_and_extract")
  dest_directory = JSONE['model_dir'][0]
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  print("maybe_download_and_extract")


def classify_image(_):
  print("classify_image")
  #JSONE = _
  maybe_download_and_extract(JSONE)
  image = (JSONE['image_file'][0] if JSONE['image_file'][0] else
           os.path.join(JSONE['model_dir'][0], 'cropped_panda.jpg'))
  run_inference_on_image(image, JSONE)
  print("classify_image")
############################################################################################

def ck(J):
  model = ResNet50(weights='imagenet')

  img_path = J['image_file'][0]
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  dp = decode_predictions(preds, top=3)[0]
  sendJson = []
  for i in range(3):
    sendJson.append({"String": dp[i][1], "Score": round(float(dp[i][2]), 4)})

  # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
  return sendJson




  
