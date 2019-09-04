# voice_recognition_ros
I created this ros package for voice recognition to use in a personal project. It uses VGGVOX keras model for creating speaker embeddings. For analysing the speech and providing the speaker intent, bot developed on wit.ai is used.

## Package details
This package contains contains the following python scripts to add, enroll or delete users:

1. **create_user_dataset.py**: to record or delete the datasets.
2. **enroll_speaker**: to create the speaker embedding  from dataset and save as pickle file
3. **listener.py**: a ros node to continously listen for voices
4. **speaker_verification.py**: a ros node to verify the speaker and connect to wit.ai for voice analysis and getting the response.

## Requirements
+ Tensorflow
+ Keras
+ librosa
+ Pyaudio
+ wit
+ scipy

## Using the package
### Adding the speaker data to the dataset
Run the following line of code and follow the instruction
```console
foo@bar:~$ python /path/to/create_user_dataset.py 
```
### Enrolling the speaker
To enroll(create embeddings) the new user run the following command with 'new' or to enroll all the speaker present in the dataset use 'full' 
```console
foo@bar:~$ python /path/to/enroll_speakers.py --add new/full
```

### Deleting the speaker
To delete the speaker embeddings and dataset run follow command in terminal. Replace speaker_name with the speaker name to delete. 
```console
foo@bar:~$ python /path/to/enroll_speakers.py --delete speaker_name
```

### Ros Speaker Verification
Do following modifications and run the launch file voice_launch.launch file.
+ Add the users name in config/users_list.yaml for which the application has to accessed.
+ Add the wit.ai access token in src/speaker_verification.py.


## References
This package contains code and data from other github users and Stackoverflow answers which are mentioned below
1. The voice recorder used in this project is taken from a [Stackeoverflow](https://stackoverflow.com/questions/18406570/python-record-audio-on-detected-sound) answer written by user Primusa.
2. Noice removal: the denoise.py script used in this package for noise removal has been written by [Tim Sainburg](https://timsainburg.com/noise-reduction-python.html).
3. The VGGVOX keras model used for creating speaker embedding is trained by [Linh Vu](https://github.com/linhdvu14/vggvox-speaker-identification) and used here.
