#!/usr/bin/python

import os
import sys
import pickle
import argparse
import numpy as np

from utils import vggvox_model, build_buckets, get_fft_spectrum

### Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--add", help="adding the speakers, values 'full' or 'new'", type=str)
parser.add_argument("--delete", help="delete the speakers, value is name of the speaker", type=str)


### Defining paths
path = os.path.dirname(os.path.abspath(__file__))
path_dataset = os.path.join(path, "../config/dataset")
path_weight = os.path.join(path, "../config/model/weights.h5")
path_embedding = os.path.join(path, "../config/speaker_embedding")

def get_embeddings(model, buckets, path):
    embeddings = []
    files = os.listdir(path)
    for file_ in files:
        x = os.path.join(path, file_)
        features = get_fft_spectrum(x, buckets)
        a, b = features.shape
        embeddings.append(np.squeeze(model.predict(features.reshape(1, a, b, 1))))
    embeddings = np.asarray(embeddings)
    embeddings = np.average(embeddings, axis=0)
    return embeddings

def main(args):
    if args.add:
        list_speaker_dataset = os.listdir(path_dataset)
        model = vggvox_model()
        model.load_weights(path_weight)
        buckets = build_buckets()

        if args.add == 'full':
            speaker_dict = {}
            for speaker in list_speaker_dataset:
                speaker_dict[speaker] = get_embeddings(model, buckets, os.path.join(path_dataset, speaker))
                 
            with open(path_embedding, 'wb') as f:
                pickle.dump(speaker_dict,f)

        elif args.add == 'new':
            with open(path_embedding, 'rb') as f:
                speaker_dict = pickle.load(f)
            speaker_list = []
            for speaker in speaker_dict.iterkeys():
                speaker_list.append(speaker)
            for speaker in list_speaker_dataset:
                if speaker in speaker_list:
                    continue
                else:
                    speaker_dict[speaker] = get_embeddings(model, buckets, os.path.join(path_dataset, speaker))
            
            with open(path_embedding, 'wb') as f:
                pickle.dump(speaker_dict,f)

    elif args.delete:
        with open(path_embedding, 'rb') as f:
            speaker_dict = pickle.load(f)
        speaker_list = []
        for speaker in speaker_dict.iterkeys():
            speaker_list.append(speaker)

        if args.delete in speaker_list:
            os.removedirs(os.path.join(path_dataset, args.delete))
            del speaker_dict[args.delete]
        
        with open(path_embedding, 'wb') as f:
            pickle.dump(speaker_dict,f)
    
    else:
        sys.exit('no argument provided')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    
