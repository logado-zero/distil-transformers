import os
import numpy as np
import csv
import six
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import faiss
import tqdm 
import tensorflow as tf

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

    
def read_word_data(input_file):
    X = []

    rows = sum(1 for _ in open(input_file, 'r'))
    with tqdm.auto.tqdm(total=rows, desc="Generate data from ...{}".format(input_file[-12:])) as bar:
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                bar.update()
                if len(line) == 0:
                    continue
                x = convert_to_unicode(line[0])
                x = " ".join(x)
                X.append(x)
    return X
    
if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()

    #required arguments
    parser.add_argument("--input_file", required=True, help="path of file want to knn")
    parser.add_argument("--output_file", required=True, help="path of result knn data")

    args = vars(parser.parse_args())
    # Read data from file
    train_examples = read_word_data(args["input_file"])
    # Create embedder
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    #Encode embedding

    corpus_embeddings = embedder.encode(train_examples, show_progress_bar= True, device = 'cuda')

    corpus_embeddings_ = np.array(corpus_embeddings)
    # Set size embedding
    d = 768
    # Prepare for knn
    nb = corpus_embeddings_.shape[0]
    nq = corpus_embeddings_.shape[0]
    xb = corpus_embeddings_
    xq = corpus_embeddings_
    # Set up faiss knn
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)

    k = 20                          # we want to see 20 nearest neighbors
    D, I = index.search(xq, k)     # actual search
    #Save index
    pickle.dump(I, open(args["output_file"], "wb" ) )   
      



