import os  # Importing the operating system module
from tkinter import *  # Importing the tkinter module for GUI
from tkinter import ttk
from tkinter import filedialog
import tensorflow as tf  # Importing TensorFlow
from tensorflow import keras
from keras import layers
import numpy as np  # Importing the NumPy library for numerical computations
import csv  # Importing the csv module for reading and writing CSV files
import requests  # Importing the requests module for sending HTTP requests
import json  # Importing the json module for working with JSON data
from jiwer import wer  # Importing the jiwer module for calculating the word error rate
import subprocess  # Importing the subprocess module for running system commands

class AudioPlayerGUI:
    def __init__(self):
        """
        Initializes the instance of the AudioPlayerGUI class with its attributes and Tkinter widgets.
        """

        # A list to store the .wav files
        self.wavs = []
        # The index of the selected .wav file in the list
        self.selected_wav = -1  

        # Create Tkinter window and set properties
        self.root = Tk()
        self.root.title("Speech To Text")
        self.root.state('zoomed')


        # Create GUI widgets

        # Create a toolbar label widget and add it to the top of the window, filling the horizontal space.
        self.toolbar = ttk.Label(self.root)
        self.toolbar.pack(side=TOP,fill=X)

        # Create a browse button widget and add it to the toolbar, aligning it to the left and adding some padding.
        self.browse_button = ttk.Button(self.toolbar, text="Browse...", command=self.browse_folder)
        self.browse_button.pack(side=LEFT, padx=(5,0))

        # Create a restore button widget and add it to the toolbar, aligning it to the left.
        self.restore_button = ttk.Button(self.toolbar, text="Restore Test Data", command=self.restore_test_data)
        self.restore_button.pack(side=LEFT)

        # Create an audio button widget and add it to the toolbar, aligning it to the left and setting its state to "disabled".
        self.audio_button = ttk.Button(self.toolbar, text="Play Audio", command=self.play_wav_file, state="disabled")
        self.audio_button.pack(side=LEFT)

        # Create a statusbar label widget and add it to the bottom of the window, filling the horizontal space and adding some padding.
        self.statusbar = ttk.Label(self.root)
        self.statusbar.pack(side=BOTTOM,fill=X, padx=5, pady=5)
        
        # Create a predict button widget and add it to the statusbar, aligning it to the left and setting its state to "disabled".
        self.predict_button = ttk.Button(self.statusbar, text="Predict", command=self.predict_file, state="disabled")
        self.predict_button.pack(side=LEFT, anchor="nw", padx=10)
        
        # Create a frame widget for the prediction section and add it to the statusbar, filling the horizontal space.
        self.prediction = ttk.Frame(self.statusbar, borderwidth=0)
        self.prediction_label = ttk.Label(self.prediction, text="Prediction:", width=10)
        self.prediction_label.pack(side=LEFT)
        self.prediction.pack(side=TOP,fill=X)
        
        # Create a StringVar variable for the prediction and create a readonly entry widget to display it.
        self.prediction_var = StringVar()
        self.prediction_entry = ttk.Entry(self.prediction, state="readonly", textvariable=self.prediction_var)
        self.prediction_entry.pack(side=TOP,fill=X, pady=(5,0))

        # Create a frame widget for the actual section and add it to the statusbar, filling the horizontal space.
        self.actual = ttk.Frame(self.statusbar)
        self.actual_label = ttk.Label(self.actual, text="Actual:", width=10)
        self.actual_label.pack(side=LEFT)
        self.actual.pack(side=TOP,fill=X)
        
        # Create a StringVar variable for the actual text and create a readonly entry widget to display it.
        self.actual_var = StringVar()
        self.actual_entry = ttk.Entry(self.actual, state="readonly", textvariable=self.actual_var)
        self.actual_entry.pack(side=TOP,fill=X, pady=(5,0))

        # Create a Frame to display the Word Error Rate (WER) and associate it with an Entry widget
        self.wer = ttk.Frame(self.statusbar, borderwidth=0)
        self.wer_var = StringVar()
        self.wer_entry = ttk.Entry(self.wer, state="readonly", textvariable=self.wer_var, width=13)
        self.wer_entry.pack(side=BOTTOM, pady=(5,0))

        # Create a Label to display the WER text and pack it to the right of the Frame
        self.wer_label = ttk.Label(self.wer, text="Word Error Rate:")
        self.wer_label.pack(side=RIGHT)

        # Pack the Frame to the right of the status bar
        self.wer.pack(side=RIGHT, fill=X)
        
        # Create a vertical scrollbar and associate it with the listbox widget
        self.scrol_y = Scrollbar(self.root,orient=VERTICAL)
        self.scrol_y.pack(side=RIGHT,fill=Y)
        self.wavs_listbox = Listbox(self.root, width=80, yscrollcommand=self.scrol_y.set, selectmode=SINGLE, font=("arial"))

        # Pack the listbox widget, expand it to fill the available space and associate the yscrollcommand with the scrollbar
        self.wavs_listbox.pack(expand=True, fill=BOTH, padx=(5,0))
        self.wavs_listbox.bind('<<ListboxSelect>>', self.on_selection_change)
        self.scrol_y.config(command=self.wavs_listbox.yview)
        
        # defining the Transcription Lookup Table and Audio Parameters

        # Define the characters that can appear in the transcription, create a lookup table for them and its inverse
        self.characters = [x for x in "abcdefghijklmnopqrstuvwxyz?!' "]
        self.char_to_num = layers.StringLookup(vocabulary=self.characters, oov_token="")
        self.num_to_char = layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)

        # Define the parameters for the audio preprocessing
        self.frame_length = 256
        self.frame_step = 160
        self.fft_length = 384

        # Call the init_wavs_list function with an empty string to initialize the list of audio files in the directory
        self.init_wavs_list('')

    
    def restore_test_data(self):
        """
        Restores test data.
        """

        self.init_wavs_list('')
        self.predict_button.config(state="disabled")


    def init_wavs_list(self, folder):
        """
        Initializes the list of wave files.
        
        Args:
            folder (str): The path to the folder containing the wave files.
        """

        # Reset variables and listbox when initializing wavs list
        self.wavs = []
        self.wavs_listbox.delete(0, END)
        self.actual_var.set("")
        self.prediction_var.set("")
        self.wer_var.set("")

        # If folder path is not provided, set the default path
        if (len(folder) == 0):
            if (os.path.exists(os.path.join(os.getcwd(), 'TestData'))):
                folder = os.path.join(os.getcwd(), 'TestData')
            else:
                self.wavs_listbox.insert(END, "Unable to find TestData Folder in directory")
                self.wavs_listbox.itemconfig(END, fg = 'red')
                self.restore_button.config(state="disabled")
                self.prediction.pack_forget()
                self.actual.pack_forget()
                self.wer.pack_forget()
                return

        self.prediction.pack(side=TOP,fill=X)
        self.actual.pack(side=TOP,fill=X)
        self.wer.pack(side=RIGHT, fill=X)

        # If metadata.csv file exists in the folder, read it and display the wave files
        if os.path.exists(os.path.join(folder, 'metadata.csv')):
            self.wavs_listbox.delete(0, END)
            with open(os.path.join(folder, 'metadata.csv'), newline='') as csvfile:
                wavsreader = csv.reader(csvfile, delimiter='|')
                i = 0
                for row in wavsreader:
                    self.wavs.append(os.path.join(folder, row[0]) + '.wav')
                    self.wavs_listbox.insert(i, row[1])
                    i = i + 1
        # If metadata.csv file doesn't exist, display all the wave files in the folder
        else:
            self.actual.pack_forget()
            self.wer.pack_forget()
            i = 0
            files = os.listdir(folder)  
            for f in files:
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.wav'):
                    self.wavs.append(os.path.join(folder, f))
                    self.wavs_listbox.insert(i, f)
                    i = i + 1

            # If no wave file found in the folder, display a message
            if len(self.wavs) == 0:
                self.wavs_listbox.insert(END, "No WAV files found in directory")
                self.wavs_listbox.itemconfig(END, fg = 'red')
                self.prediction.pack_forget()
                self.actual.pack_forget()
                self.wer.pack_forget()
                return
    

    def browse_folder(self):
        """
        Opens a dialog to select a folder, then initializes the list of audio files in the selected folder.
        """

        # Use Tkinter file dialog to get audio file path
        folder = filedialog.askdirectory()

        # If a folder has been selected, initialize the list of WAV files in the UI
        if len(folder) > 0:
            self.init_wavs_list(folder)


    def on_selection_change(self, event):
        """
        Updates the GUI when the user selects a new WAV file from the listbox.

        Args:
            event (tkinter.Event): The selection event.
        """

        w = event.widget
        self.selected_wav = int(w.curselection()[0])

        # Clear previous transcription and WER values
        self.prediction_var.set("")
        self.actual_entry.delete(0, END)
        self.wer_var.set("")

        # If the selected WAV file has a different transcription than the one currently displayed, update the transcription field
        if (not self.wavs[self.selected_wav].lower().endswith(w.get(w.curselection()[0]).lower())):
            self.actual_var.set(w.get(w.curselection()[0]))

        # Enable the prediction and play buttons
        self.predict_button.config(state="normal")
        self.audio_button.config(state="normal")

    def play_wav_file(self):
        """
        Plays the selected audio file using the Windows Media Player.
        """
        
        subprocess.call(["C:\Program Files (x86)\Windows Media Player\wmplayer", self.wavs[self.selected_wav]])


    def predict_file(self):
        """
        Predicts the text transcript of a selected WAV file using a pre-trained machine learning model.
        It sends an HTTP request to a speech recognition service using the WAV file's encoded data, and receives a prediction in JSON format. 
        The predicted text is then decoded and displayed in the GUI, along with the calculated word error rate (WER).
        """

        # Get the index of the selected audio file
        index = self.selected_wav
        if (index == -1):
            return

        # Clear the prediction entry field
        self.prediction_entry.delete(0, END)

        # Encode the audio file and prepare it for prediction
        X = tf.expand_dims(self.encode_single_sample(self.wavs[index]), axis = 0) 

        # Convert the audio data to JSON format for sending to the prediction service
        data = X.numpy()
        json_data = json.dumps({'signature_name': 'serving_default', 'instances': data.tolist()})  

        # Set the HTTP request headers
        headers = {'content-type': 'application/json'}

        # Set the URL for the prediction service
        predict_service_url = 'http://13.95.7.205/v1/models/speech2text:predict'

        # Make a POST request to the prediction service with the audio data in JSON format
        json_response = requests.post(predict_service_url, data=json_data, headers=headers)

        # Decode the prediction from the JSON response
        prediction = json.loads(json_response.text)['predictions']
        prediction_decoded = self.decode_single_prediction(np.array(prediction))

        # Display the prediction in the GUI
        self.prediction_var.set(prediction_decoded[0])

        # Calculate and display the WER score
        self.wer_var.set(format(wer(self.actual_var.get().lower(), self.prediction_var.get()), '.2f'))


    def encode_single_sample(self, wav_file):
        """
        Encodes a single WAV file for input to the model.
        
        Args:
            wav_file (str): The path to the WAV file to be encoded.
            
        Returns:
            spectogram (Tensor): The encoded spectogram of the input WAV file.
        """

        # Read the contents of the WAV file
        file = tf.io.read_file(wav_file)
        
        # Decode the WAV file to a 1-dimensional tensor
        audio, _ = tf.audio.decode_wav(file, desired_channels=1)
        
        # Squeeze the tensor to remove the channel dimension and cast it to float32
        audio = tf.squeeze(audio, axis=1)
        audio = tf.cast(audio, tf.float32)
        
        # Compute the Short-Time Fourier Transform (STFT) of the audio
        spectogram = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)
        
        # Convert the complex spectrogram to magnitude spectrogram
        spectogram = tf.abs(spectogram)
        
        # Apply a power transformation to the spectrogram
        spectogram = tf.math.pow(spectogram, 0.5)
        
        # Compute the mean and standard deviation of the spectrogram across the frequency axis
        means = tf.math.reduce_mean(spectogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectogram, 1, keepdims=True)
        
        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation
        spectogram = (spectogram - means) / (stddevs + 1e-10)

        return spectogram


    def decode_single_prediction(self, pred):
        """
        Decodes a single model prediction into a human-readable string.
        
        Args:
            pred (ndarray): A model prediction, as a numpy ndarray.
            
        Returns:
            output_text (list of str): A list of human-readable strings representing the prediction.
        """

        # Compute the length of the input sequence for CTC decoding
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        # Decode the prediction using CTC decoding
        ctc = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        results = ctc[0][0]

        # Convert the decoded indices to characters and join them to form the output text
        output_text = []
        for result in results:
            # Convert the indices to characters
            result = tf.strings.reduce_join(self.num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        
        return output_text


    def run(self):
        """
        Runs the Tkinter event loop, allowing the application to listen for and respond to user input events.
        """

        self.root.mainloop()
    

if __name__ == "__main__":
    # Create an instance of the AudioPlayerGUI class
    gui = AudioPlayerGUI()
    # Run the GUI by calling the run method of the AudioPlayerGUI instance
    gui.run()





