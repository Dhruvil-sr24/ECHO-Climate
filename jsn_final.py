import torch
import requests
import inflect
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline, 
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
from datasets import load_dataset
import soundfile as sf
# import librosa
# -----------
import serial
import wave
import struct
import time
# import wave
# import struct
import numpy as np
from scipy.signal import resample

# ---------------
def setup_serial_communication () :
	ser = serial.Serial(port="COM7" , baudrate=1000000)
	ser.reset_input_buffer()
	ser.reset_output_buffer()
	return ser

def connect_to_esp32 (ser) :
	while ser.read() != b'\x01' :
		pass
	r_data_1 = int.from_bytes(ser.read(),byteorder="little")
	r_data_2 = int.from_bytes(ser.read(),byteorder="little")
	r_data_3 = int.from_bytes(ser.read(),byteorder="little")
	if (r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3) :
		ser.write(bytes([3]))
		ser.write(bytes([2]))
		ser.write(bytes([1]))
	r_data_1 = int.from_bytes(ser.read(),byteorder="little")
	r_data_2 = int.from_bytes(ser.read(),byteorder="little")
	r_data_3 = int.from_bytes(ser.read(),byteorder="little")
	if (r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3) :
		print("Connected To The ESP32 ......")
		return
	while True :
		pass

def recieve_data_from_esp32(ser):
    # Wait for synchronization byte
    while ser.read() != b'\x01':
        pass

    # Read the synchronization bytes
    r_data_1 = int.from_bytes(ser.read(), byteorder="little")
    r_data_2 = int.from_bytes(ser.read(), byteorder="little")
    r_data_3 = int.from_bytes(ser.read(), byteorder="little")

    # If sync matches, send acknowledgment
    if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
        ser.write(bytes([3]))
        ser.write(bytes([2]))
        ser.write(bytes([1]))

    # Prepare a list to store the received 16-bit signed integers
    audio_data = []

    # Receive 96000 bytes, which represent 48000 samples (16-bit each sample)
    for _ in range(48000):  # 48000 samples, each consisting of 2 bytes (MSB, LSB)
        # Read the MSB and LSB
        msb = ser.read(1)
        lsb = ser.read(1)

        # Reassemble the 16-bit signed integer
        if msb and lsb:
            sample = (int.from_bytes(msb, byteorder='big', signed=False) << 8) | int.from_bytes(lsb, byteorder='big', signed=False)

            # Convert it back to signed 16-bit integer if necessary
            if sample >= 0x8000:
                sample -= 0x10000

            # Append to the audio data list
            audio_data.append(sample)

    # Check for end-of-transmission synchronization bytes
    r_data_1 = int.from_bytes(ser.read(), byteorder="little")
    r_data_2 = int.from_bytes(ser.read(), byteorder="little")
    r_data_3 = int.from_bytes(ser.read(), byteorder="little")

    # Verify the end synchronization
    if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
        print("Data Received Of Length:", len(audio_data))
        print("Data Reception Complete ......")
        return audio_data  # Return the list of 16-bit signed samples

    # If sync doesn't match, hang in an infinite loop
    while True:
        pass

def transfer_data_to_esp32(ser, audio_data):
    # Send the synchronization bytes to start the transmission
    ser.write(bytes([3]))
    ser.write(bytes([2]))
    ser.write(bytes([1]))

    # Wait for synchronization response from ESP32
    while ser.read() != b'\x01':
        pass

    # Read synchronization acknowledgment from ESP32
    r_data_1 = int.from_bytes(ser.read(), byteorder="little")
    r_data_2 = int.from_bytes(ser.read(), byteorder="little")
    r_data_3 = int.from_bytes(ser.read(), byteorder="little")

    # If synchronization is successful
    if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
        # Loop through each audio sample in the list (16-bit signed integers)
        for sample in audio_data:
            # Break down the 16-bit signed integer sample into MSB and LSB
            msb = (sample >> 8) & 0xFF   # Extract Most Significant Byte (MSB)
            lsb = sample & 0xFF          # Extract Least Significant Byte (LSB)

            # Transmit the MSB first, followed by the LSB
            ser.write(bytes([msb]))
            ser.write(bytes([lsb]))

        # Send end of transmission synchronization bytes
        ser.write(bytes([3]))
        ser.write(bytes([2]))
        ser.write(bytes([1]))

        # Wait for final acknowledgment from ESP32
        while ser.read() != b'\x01':
            pass
        r_data_1 = int.from_bytes(ser.read(), byteorder="little")
        r_data_2 = int.from_bytes(ser.read(), byteorder="little")
        r_data_3 = int.from_bytes(ser.read(), byteorder="little")

        # If final acknowledgment is received, confirm data transfer completion
        if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
            print("Data Transfer Completed! ......")
            return

    # If something goes wrong, hang in an infinite loop
    while True:
        pass


def recieve_sensor_data_from_esp32(ser):
    # Wait for synchronization bytes
    while ser.read() != b'\x01':
        pass
    if ser.read() == b'\x02' and ser.read() == b'\x03':
        # Send acknowledgment
        ser.write(bytes([3]))
        ser.write(bytes([2]))
        ser.write(bytes([1]))

        # Receive the 2 bytes of temperature (MSB, LSB)
        temp_msb = ser.read(1)
        temp_lsb = ser.read(1)

        # Reassemble the temperature (16-bit signed integer)
        if temp_msb and temp_lsb:
            temp = (int.from_bytes(temp_msb, byteorder='big', signed=False) << 8) | int.from_bytes(temp_lsb, byteorder='big', signed=False)
            # Convert it back to signed 16-bit integer if necessary
            if temp >= 0x8000:
                temp -= 0x10000

        # Receive the 2 bytes of humidity (MSB, LSB)
        hum_msb = ser.read(1)
        hum_lsb = ser.read(1)

        # Reassemble the humidity (16-bit signed integer)
        if hum_msb and hum_lsb:
            hum = (int.from_bytes(hum_msb, byteorder='big', signed=False) << 8) | int.from_bytes(hum_lsb, byteorder='big', signed=False)
            # Convert it back to signed 16-bit integer if necessary
            if hum >= 0x8000:
                hum -= 0x10000

        # Check for end-of-transmission synchronization bytes
        r_data_1 = int.from_bytes(ser.read(), byteorder="little")
        r_data_2 = int.from_bytes(ser.read(), byteorder="little")
        r_data_3 = int.from_bytes(ser.read(), byteorder="little")

        # Check for end-of-transfer synchronization bytes
        # Verify the end synchronization
        if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
            print(f"Temperature: {temp}°C, Humidity: {hum}%")
            # l=[]
            return [temp, hum]

    # If synchronization fails, hang in an infinite loop
    while True:
        pass


def transfer_control_signal(ser , control_signal) :
     # Send the synchronization bytes to start the transmission
    ser.write(bytes([3]))
    ser.write(bytes([2]))
    ser.write(bytes([1]))

    # Wait for synchronization response from ESP32
    while ser.read() != b'\x01':
        pass

    # Read synchronization acknowledgment from ESP32
    r_data_1 = int.from_bytes(ser.read(), byteorder="little")
    r_data_2 = int.from_bytes(ser.read(), byteorder="little")
    r_data_3 = int.from_bytes(ser.read(), byteorder="little")

    # If synchronization is successful
    if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:

		# Sending control signal
        if (control_signal) :
            ser.write(bytes([1]))
            ser.write(bytes([2]))
            ser.write(bytes([3]))
        else :
            ser.write(bytes([3]))
            ser.write(bytes([2]))
            ser.write(bytes([1]))

        # Send end of transmission synchronization bytes
        ser.write(bytes([3]))
        ser.write(bytes([2]))
        ser.write(bytes([1]))

        # Wait for final acknowledgment from ESP32
        while ser.read() != b'\x01':
            pass
        r_data_1 = int.from_bytes(ser.read(), byteorder="little")
        r_data_2 = int.from_bytes(ser.read(), byteorder="little")
        r_data_3 = int.from_bytes(ser.read(), byteorder="little")

        # If final acknowledgment is received, confirm data transfer completion
        if r_data_1 == 1 and r_data_2 == 2 and r_data_3 == 3:
            print("Signal Transfer Completed! ......")
            if control_signal == True :
                return recieve_sensor_data_from_esp32(ser)
            else :
                return

    # If something goes wrong, hang in an infinite loop
    while True:
        pass


def raw_to_wav(filename, audio_data, sample_rate=16000):
    # Open a wave file
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: (channels, sample width, sample rate, number of frames, compression type, compression name)
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 2 bytes (16-bit audio)
        wav_file.setframerate(sample_rate)  # Sample rate in Hz

        # Convert the list of audio samples (16-bit signed integers) into a byte string
        for sample in audio_data:
            # Pack each sample into binary format ('h' stands for 16-bit signed int)
            wav_file.writeframes(struct.pack('<h', sample))

# def wav_to_raw(filename):
#     # Open the wave file for reading
#     with wave.open(filename, 'r') as wav_file:
#         # Verify the file parameters
#         num_channels = wav_file.getnchannels()    # Should be 1 (mono)
#         sample_width = wav_file.getsampwidth()    # Should be 2 (16-bit audio)
#         sample_rate = wav_file.getframerate()     # Should be 16000 Hz
#         num_frames = wav_file.getnframes()        # Total number of frames (samples)

#         print(f"Channels: {num_channels}")
#         print(f"Sample Width: {sample_width} bytes")
#         print(f"Sample Rate: {sample_rate} Hz")
#         print(f"Number of Frames: {num_frames}")

#         if num_channels != 1 or sample_width != 2 or sample_rate != 16000:
#             # raise ValueError("Unsupported WAV format. Expected 16-bit mono audio at 16000 Hz.")
#             print('error')
#         # Read the entire data from the WAV file
#         raw_data = wav_file.readframes(num_frames)

#         # Unpack the byte data into 16-bit signed integers
#         audio_data = list(struct.unpack('<' + 'h' * num_frames, raw_data))

#     return audio_data

def wav_to_raw(filename):
    target_frames = 48000  # The target number of frames
    target_sample_rate = 16000  # The target sample rate (if you want resampling)
    target_channels = 1  # Mono audio
    target_sample_width = 2  # 16-bit (2 bytes) per sample
    
    # Open the wave file for reading
    with wave.open(filename, 'r') as wav_file:
        # Get file parameters
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()    # Bytes per sample
        sample_rate = wav_file.getframerate()     # Samples per second
        num_frames = wav_file.getnframes()        # Total number of frames
        
        print(f"Channels: {num_channels}")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Number of Frames: {num_frames}")
        
        # Read the data from the WAV file
        raw_data = wav_file.readframes(num_frames)
        
        # Convert the raw bytes into an appropriate NumPy array for manipulation
        if sample_width == 1:  # 8-bit audio, unsigned
            audio_data = np.frombuffer(raw_data, dtype=np.uint8) - 128  # Convert unsigned 8-bit to signed
        elif sample_width == 2:  # 16-bit audio, signed
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
        else:
            raise ValueError("Unsupported sample width. Only 8-bit and 16-bit audio are supported.")
        
        # If the WAV file has more than one channel, downmix to mono
        if num_channels > 1:
            audio_data = np.mean(audio_data.reshape(-1, num_channels), axis=1).astype(np.int16)
        
        # If the sample rate is different from the target, resample the audio
        if sample_rate != target_sample_rate:
            num_target_frames = int((target_frames / target_sample_rate) * sample_rate)
            audio_data = resample(audio_data, num_target_frames)
            audio_data = audio_data.astype(np.int16)
        
        # Trim or pad to 48,000 frames
        if len(audio_data) > target_frames:
            # Trim the audio data if there are more than 48,000 frames
            audio_data = audio_data[:target_frames]
            print(f"Trimmed to {target_frames} frames.")
        elif len(audio_data) < target_frames:
            # If there are fewer than 48,000 frames, pad with silence (zeros)
            padding_needed = target_frames - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_needed), 'constant', constant_values=0)
            print(f"Padded with {padding_needed} frames of silence.")
    
    print("Done")
    return audio_data.tolist()  # Convert to list if needed

# -----------------------------------------------------------------------------------------------------------


# Set device and data type for optimized performance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models once and reuse them for efficiency
def load_models():
    # Load the speech-to-text model (Whisper)
    model_id = "openai/whisper-tiny.en"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create a pipeline for automatic speech recognition (ASR)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,  # Use mixed precision for faster processing
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )
    
    # Load text-to-speech (TTS) models
    processor_tts = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    return pipe, processor_tts, model_tts, vocoder

# Optimized speech-to-text processing with mixed precision
def process_sample(pipe, audio_path):
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):  # Mixed precision
        result = pipe(audio_path)  # Perform speech recognition
    return result['text']

# API Call to get city temperature (optimized)
def get_city_temperature(city):
    api_key = "e7777b4e947b1311b4d5fbe5a004be21"
    url = f"http://api.openw0eathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Use inflect engine to convert numbers to words
        p = inflect.engine()
        txt_temp = p.number_to_words(temperature)
        humidity_temp = p.number_to_words(humidity)
        
        return f"{txt_temp} celsius and {humidity_temp} percent."
    else:
        return "i couldn't hear that. please say again"

# Text-to-speech function
def generate_speech(text, processor_tts, model_tts, vocoder):
    # Prepare text inputs
    inputs = processor_tts(text=text, return_tensors="pt").to(device)
    
    # Load speaker embeddings only once to avoid reloading
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
    
    # Generate speech
    with torch.no_grad():  # Disable gradients for faster inference
        speech = model_tts.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save speech to file
    sf.write("speech.wav", speech.cpu().numpy(), samplerate=16000)






# Main function to handle speech-to-text, get data, and text-to-speech
def main():

    

    pipe, processor_tts, model_tts, vocoder = load_models()
    # +++++++++++++++++++++++++++++++++
    ser = setup_serial_communication()
    connect_to_esp32(ser)
    while True :
        audio_data = recieve_data_from_esp32(ser)
        print(f"First 5 audio samples: {audio_data[:5]}")
        raw_to_wav('output_audio.wav', audio_data)
        # +++++++++++++++++++++++++++++++++
        # Process audio input for speech-to-text
        sample = "output_audio.wav"
        jsn_ot = process_sample(pipe, sample)
        print(f"Transcribed Text: {jsn_ot}")
        
        # Extract city name and get temperature data
        jsn_ot_list=jsn_ot.split()
        print(jsn_ot_list)
        if 'now?' in jsn_ot_list :
            a=transfer_control_signal(ser ,True)
            p=inflect.engine()
            txt_temp_dht = p.number_to_words(a[0])
            humidity_temp_dht = p.number_to_words(a[1])
            city_temperature_humid=f"{txt_temp_dht} celcius and {humidity_temp_dht} p"
        else:
            a=transfer_control_signal(ser,False)
            jsn_ot_api = jsn_ot
            words = jsn_ot_api.split()
            city = words[-1]  # Assuming last word is the city
            # print(city[:-1])
            city_temperature_humid = get_city_temperature(city[:-1])#changes city ->>>> city[:-1]
            # a=transfer_control_signal(ser ,False)
        
        # Prepare the final text for TTS
        final_text = f"{city_temperature_humid}"
        print(f"Final Text for TTS: {final_text}")
        
        # Generate speech from text
        generate_speech(final_text, processor_tts, model_tts, vocoder)
        print("Speech generated and saved as 'speech.wav'")
        audio_data = wav_to_raw('speech.wav')
        transfer_data_to_esp32(ser,audio_data)
        time.sleep(5)
        # int16_speech_array = generate_speech(final_text, processor_tts, model_tts, vocoder)
        # print(f"Generated signed 16-bit integer array for ESP32.{int16_speech_array}",)



# Execute the main function
if __name__ == "__main__":
    main()
