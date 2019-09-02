#!/usr/bin/python

import os
import yaml
import json
import pickle
import numpy as np
from wit import Wit
import tensorflow as tf
from scipy.spatial.distance import cosine

import constants as c
from utils import vggvox_model, build_buckets, get_fft_spectrum

import rospkg
import rospy
from std_msgs.msg import String

### defining paths
path = os.path.dirname(os.path.abspath(__file__))
path_weight = os.path.join(path, "../config/model/weights.h5")
path_embedding = os.path.join(path, "../config/speaker_embedding")
path_user_list = os.path.join(path, '../config/users_list.yaml')
path_audio = os.path.join(path, '../config/audio.wav')

class VGGVOX:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        self.model = self.load()

    def load(self):
        with self.graph.as_default():
            with self.session.as_default():
                model = vggvox_model()
                model.load_weights(path_weight)
        return model

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

class voice_analysis:
    def __init__(self):
        self.model = VGGVOX()
        self.buckets = build_buckets()
        self.client = Wit(access_token="ADD THE WIT ACCESS TOKEN")
    
        with open(path_user_list, 'rb') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.users_list = data['users']
        
        with open(path_embedding, 'rb') as f:
            self.speaker_dict = pickle.load(f)

        sub_audio = rospy.Subscriber('audio_file', String, self.callback)
        self.pub_res = rospy.Publisher('wit_response', String, queue_size=1)

    def callback(self, file_path):
        #path_audio = file_path.data

        if self.verify_speaker(path_audio):
            with open(path_audio, 'rb') as f:
                result = self.client.speech(f, None, {'Content-Type': 'audio/wav'})
            wit_response = json.dumps(result) ### converting the dictionary to string
            print(result)
            self.pub_res.publish(wit_response)

    def verify_speaker(self, path_audio):
        verified = False

        features = get_fft_spectrum(path_audio, self.buckets)
        h, w = features.shape
        test_embedding = np.squeeze(self.model.predict(features.reshape(1, h, w, 1)))

        speakers = []
        scores = []
        for speaker, embedding in self.speaker_dict.items():
            dist = cosine(test_embedding, embedding)
            scores.append(1./dist)
            speakers.append(speaker)

        sigmoid_score = np.exp(scores)/np.sum(np.exp(scores))
        idx = np.argmax(sigmoid_score)

        score_threshold = 0.8 #(1.0/len(self.speaker_dict))*1.8
        print(speakers, sigmoid_score, score_threshold)
        if (speakers[idx] in self.users_list) and (sigmoid_score[idx] > score_threshold):
            verified = True
        return verified


def main():
    rospy.init_node('vioce_analysis')
    va = voice_analysis()
    rospy.spin()

if __name__ == "__main__":
    main()