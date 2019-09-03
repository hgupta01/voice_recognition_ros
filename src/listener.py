#!/usr/bin/python

import os
import wave
import time
import math
import pyaudio
import struct

from utils import load_wav
from denoise import removeNoise

import rospy
import rospkg
from std_msgs.msg import String

rospack = rospkg.RosPack()
audio_filename = os.path.join(rospack.get_path('voice_analysis'), 'config/audio2.wav')
noise_filename = os.path.join(rospack.get_path('voice_analysis'), 'config/noise.wav')

Threshold = 40
SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2
TIMEOUT_LENGTH = 1


def record_noise():
    print('recording noise')
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              output=True,
                              frames_per_buffer=chunk)
    
    rec = []
    current = time.time()
    end = time.time() + 1 # recording noise for 1 sec
    
    while current <= end:
        data = stream.read(chunk)
        current = time.time()
        rec.append(data)
    
    wf = wave.open(noise_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(rec))
    wf.close()
    print('recording noise completed')

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)
        self.pub_record = rospy.Publisher('audio_file', String, queue_size=1)

    def record(self, input):
        rec = []
        rec.append(input)
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        wf = wave.open(audio_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        self.pub_record.publish(audio_filename)

    def listen(self):
        print('Start listening')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record(input)

    def initiate_shutdown(self):
	self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    rospy.init_node('audio_listener')
    record_noise()
    record = Recorder()
    record.listen()
    rospy.on_shutdown(record.initiate_shutdown())
