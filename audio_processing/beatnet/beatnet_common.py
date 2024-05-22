# URL:https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/common.py
# No modification is made to the code.

# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod

import numpy as np
import librosa


class FeatureModule(object):
    """
    Implements a generic music feature extraction module wrapper.
    """

    def __init__(self, sample_rate, hop_length, num_channels=1, decibels=True):
        """
        Initialize parameters common to all feature extraction modules.

        Parameters
        ----------
        sample_rate : int or float
          Presumed sampling rate for all audio
        hop_length : int or float
          Number of samples between feature frames
        num_channels : int
          Number of independent feature channels
        decibels : bool
          Convert features to decibel (dB) units
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_channels = num_channels
        self.decibels = decibels


    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # Simply the number of hops plus one
        num_frames = 1 + len(audio) // self.hop_length

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce features
        with a given number of frames.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        # Calculate the boundaries
        max_samples = num_frames * self.hop_length - 1
        min_samples = max(1, max_samples - self.hop_length + 1)

        # Construct an array ranging between the minimum and maximum number of samples
        sample_range = np.arange(min_samples, max_samples + 1)

        return sample_range

    @abstractmethod
    def process_audio(self, audio):
        """
        Get features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio
        """

        return NotImplementedError

    def to_decibels(self, feats):
        """
        Convert features to decibels (dB) units.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        feats : ndarray
          Calculated amplitude features

        Returns
        ----------
        feats : ndarray
          Calculated features in decibels
        """

        # Simply use the appropriate librosa function
        feats = librosa.core.amplitude_to_db(feats, ref=np.max)

        return feats

    def post_proc(self, feats):
        """
        Perform post-processing steps.

        Parameters
        ----------
        feats : ndarray
          Calculated features

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        if self.decibels:
            # Convert to decibels (dB)
            feats = self.to_decibels(feats)

            # TODO - make additional variable for 0/1 scaling
            # Assuming range of -80 to 0 dB, scale between 0 and 1
            feats = feats / 80
            feats = feats + 1
        else:
            # TODO - should anything be done here? - would I ever not want decibels?
            pass

        # Add a channel dimension
        feats = np.expand_dims(feats, axis=0)

        return feats

    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with frame.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Determine the number of frames we will get
        num_frames = self.get_expected_frames(audio)

        frame_idcs = np.arange(num_frames + 1)
        # Obtain the time of the first sample of each frame
        times = librosa.frames_to_time(frames=frame_idcs,
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)

        return times

    def get_sample_rate(self):
        """
        Helper function to access sampling rate.

        Returns
        ----------
        sample_rate : int or float
          Presumed sampling rate for all audio
        """

        sample_rate = self.sample_rate

        return sample_rate

    def get_hop_length(self):
        """
        Helper function to access hop length.

        Returns
        ----------
        hop_length : int or float
          Number of samples between feature frames
        """

        hop_length = self.hop_length

        return hop_length

    def get_num_channels(self):
        """
        Helper function to access number of feature channels.

        Returns
        ----------
        num_channels : int
          Number of independent feature channels
        """

        num_channels = self.num_channels

        return num_channels

    @classmethod
    def features_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the module.

        Returns
        ----------
        tag : str
          Name of the child class calling the function
        """

        return cls.__name__