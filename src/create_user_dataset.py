#!/usr/bin/python

import sys
import wave
import time
import os
import pyaudio
import math
import struct
import pickle

import constants as c

path_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/dataset")

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / c.swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * c.SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.wav_file = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=c.FORMAT,
                                  channels=c.CHANNELS,
                                  rate=c.SAMPLE_RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=c.chunk)
        

    def record(self, input):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + c.TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(c.chunk)
            if self.rms(data) >= c.Threshold: end = time.time() + c.TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        filename = os.path.join(self.path, str(self.count)+'.wav')

        wf = wave.open(filename, 'wb')
        wf.setnchannels(c.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(c.FORMAT))
        wf.setframerate(c.SAMPLE_RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written file: {}'.format(filename))
        self.count += 1
        print('Returning to listening')


    def listen(self):
        print('Read 15 sentences from provided text with 2 second gaps in between.')
        print('Listening begin')
        self.count = 1
        while self.count<=15:
            input = self.stream.read(c.chunk)
            rms_val = self.rms(input)
            if rms_val > c.Threshold:
                self.record(input)
        print("Recording Completed")
        
def main():
    char='y'
    while (char=='y'):
        print('New user creation:starting')
        name = raw_input('Enter the name of person to be added: ')
        list_user = os.listdir(path_dataset)
        path_user = os.path.join(path_dataset, name)
        if name in list_user:
            answer = raw_input("User already exists. Press 'y' to overwrite dataset and 'n' to quit: ")
            if answer=='y' or answer=='Y':
                a = Recorder(name, path_user)
                print ("Initializing Recorder")
            else:
                sys.exit("No user datset added")
        else:
            os.mkdir(path_user)
            a = Recorder(name, path_user)
            print ("Initializing Recorder")
        
        a.listen()
        char = raw_input("Would you like to add other users. Press 'y' for yes.")


if __name__ == "__main__":
    main()
