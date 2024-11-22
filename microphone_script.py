import serial
import wave

# Define the byte values for acknowledgments
READY_ACK = 0x01
DONE_ACK = 0x02

def initialize_serial(port, baudrate, timeout):
    """Initialize the serial connection."""
    ser = serial.Serial(port, baudrate, timeout=timeout)
    return ser

def wait_for_acknowledgment(ser, expected_ack):
    """Wait for a specific acknowledgment byte from ESP32."""
    print(f"Waiting for acknowledgment {expected_ack} from ESP32...")
    while True:
        if ser.in_waiting > 0:
            ack = ser.read(1)  # Read 1 byte
            if ack == bytes([expected_ack]):
                print(f"Received acknowledgment {expected_ack} from ESP32")
                break

def send_acknowledgment(ser, ack):
    """Send an acknowledgment byte to ESP32."""
    ser.write(bytes([ack]))
    print(f"Sent acknowledgment {ack} to ESP32")

def receive_audio_data(ser, total_bytes):
    """Receive audio data from ESP32."""
    audio_data = bytearray()
    received_bytes = 0
    print("Receiving audio data...")

    while True:
        if ser.in_waiting > 0:
            # Read data until DONE_ACK or expected bytes are received
            while ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                audio_data.extend(chunk)
                received_bytes += len(chunk)

            # Check for DONE_ACK
            if received_bytes >= total_bytes and DONE_ACK in audio_data:
                print("Received DONE_ACK, data transfer complete")
                break

    return audio_data

def convert_to_wav(audio_data, wav_file_path, sample_rate):
    """Convert raw audio data to a .wav file directly."""
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)  # Sample rate

        # Write the raw audio data to the .wav file
        wav_file.writeframes(audio_data)

    print(f"Audio saved as '{wav_file_path}'")

def main():
    # Configuration
    port = 'COM7'  # Replace with your port
    baudrate = 1000000
    timeout = 5
    total_bytes = 96000  # 3 seconds of 16-bit 16kHz audio
    sample_rate = 16000
    wav_file_path = 'recorded_audio.wav'

    # Initialize serial connection
    ser = initialize_serial(port, baudrate, timeout)

    # Synchronize with ESP32
    wait_for_acknowledgment(ser, READY_ACK)
    send_acknowledgment(ser, READY_ACK)

    # Receive and save audio data
    audio_data = receive_audio_data(ser, total_bytes)

    # Convert to WAV file
    convert_to_wav(audio_data, wav_file_path, sample_rate)

    # Close serial connection
    ser.close()

if __name__ == "__main__":
    main()
