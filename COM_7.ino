#include <driver/i2s.h>
#include <DHT.h>

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#define I2S_SAMPLE_RATE 16000          // Sampling rate
#define I2S_BCLK_PIN 21                // Bit Clock pin
#define I2S_LRCK_PIN 22                // Word Select (LRCK) pin
#define I2S_DIN_PIN 19                 // Data input pin

#define I2S_WS 25                                 // Word Select (LRCK)
#define I2S_SD 32                                 // Serial Data (DIN)
#define I2S_SCK 26                                // Bit Clock (BCLK)
#define SAMPLE_RATE 16000                         // Sample rate in Hz
#define RECORD_TIME 3                             // Record time in seconds
#define BUFFER_SIZE (SAMPLE_RATE * RECORD_TIME)   // Total number of samples to record for 3 seconds
int16_t audio_buffer[BUFFER_SIZE];                // Buffer to store the 3-second recording

#define DHTPIN 15               // GPIO15 for DHT22 data line
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void setupI2S_speaker() {
  i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_TX),
      .sample_rate = I2S_SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 64,
      .use_apll = false,
      .tx_desc_auto_clear = true
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_BCLK_PIN,
      .ws_io_num = I2S_LRCK_PIN,
      .data_out_num = I2S_DIN_PIN,
      .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void play_beep(int frequency, int duration_ms) {
  int32_t samples_per_wave = I2S_SAMPLE_RATE / frequency;
  int16_t samples[samples_per_wave];
  int32_t total_samples = (I2S_SAMPLE_RATE * duration_ms) / 1000;
  for (int32_t i=0 ; i<samples_per_wave ; i++) {
    samples[i] = (i < samples_per_wave / 2) ? 32767 : -32767;
  }
  for (int32_t i=0 ; i<total_samples ; i+=samples_per_wave) {
    size_t bytes_written;
    i2s_write(I2S_NUM_0, samples, sizeof(samples), &bytes_written, portMAX_DELAY);
  }
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void play_audio() {
    size_t bytes_written;

    // Loop through the audio_buffer and send data to the I2S peripheral
    for (int i = 0; i < BUFFER_SIZE; i++) {
        // Write the 16-bit signed audio sample to the I2S peripheral
        i2s_write(I2S_NUM_0, (char*)&audio_buffer[i], sizeof(int16_t), &bytes_written, portMAX_DELAY);
    }
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void setupI2S_microphone() {

  i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 64
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_1, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_1, &pin_config);

}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void record_audio() {

  // Initializing variables
  size_t bytes_read;
  int16_t sample;
  int32_t i=0;

  // Starting beep
  play_beep(1000 , 300) ;
  delay(200) ;

  // Actual recording logic
  for (i ; i<BUFFER_SIZE ; i++) {
    i2s_read(I2S_NUM_1, &sample, sizeof(sample), &bytes_read, portMAX_DELAY);
    audio_buffer[i] = sample;
  }

  // Ending beep
  play_beep(1000 , 300) ;
  delay(200) ;

}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void setup_serial_communication () {
  Serial.begin (1000000) ;
  Serial.flush () ;
  while (Serial.available()) Serial.read() ;
  return ;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void connect_to_laptop () {
  byte s_data_1 = 0x01 ;
  byte s_data_2 = 0x02 ;
  byte s_data_3 = 0x03 ;
  Serial.write (s_data_1) ;
  Serial.write (s_data_1) ;
  Serial.write (s_data_2) ;
  Serial.write (s_data_3) ;
  while (Serial.available() <= 0) ;
  byte r_data_3 = (byte)Serial.read() ;
  while (Serial.available() <= 0) ;
  byte r_data_2 = (byte)Serial.read() ;
  while (Serial.available() <= 0) ;
  byte r_data_1 = (byte)Serial.read() ;
  if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
    Serial.write(s_data_1) ;
    Serial.write(s_data_2) ;
    Serial.write(s_data_3) ;
    return ;
  }
  while (true) ;
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void transfer_data_to_laptop() {
  // Bytes to check synchronization
  byte s_data_1 = 0x01;
  byte s_data_2 = 0x02;
  byte s_data_3 = 0x03;
  
  // Write synchronization bytes to the laptop
  Serial.write(s_data_1) ;
  Serial.write(s_data_1);
  Serial.write(s_data_2);
  Serial.write(s_data_3);

  // Wait for acknowledgment from the laptop
  while (Serial.available() <= 0);
  byte r_data_3 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_2 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_1 = (byte)Serial.read();

  // Check if the acknowledgment matches expected values
  if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
    // Transfer the 16-bit audio data as two 8-bit segments per sample
    for (int i = 0; i < BUFFER_SIZE; i++) {
      int16_t sample = audio_buffer[i];

      // Extract the MSB and LSB from the 16-bit sample
      byte msb = (sample >> 8) & 0xFF;  // Most significant byte
      byte lsb = sample & 0xFF;         // Least significant byte

      // Write the MSB first, followed by the LSB
      Serial.write(msb);
      Serial.write(lsb);
    }

    // Send the end of transfer signal
    Serial.write(s_data_1);
    Serial.write(s_data_2);
    Serial.write(s_data_3);

    return;
  }

  // If acknowledgment doesn't match, halt
  while (true);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void recieve_data_from_laptop() {
  byte s_data_1 = 0x01;
  byte s_data_2 = 0x02;
  byte s_data_3 = 0x03;
  
  // Wait for initial synchronization bytes from the laptop
  while (Serial.available() <= 0);
  byte r_data_3 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_2 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_1 = (byte)Serial.read();

  // Check if synchronization bytes match
  if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
    // Send acknowledgment bytes back to the laptop
    Serial.write(s_data_1);
    Serial.write(s_data_1);
    Serial.write(s_data_2);
    Serial.write(s_data_3);

    // Buffer to store 48000 audio samples (16-bit signed integers)
    int index = 0;

    // Receive the audio data from the laptop (MSB and LSB for each 16-bit sample)
    for (int i = 0; i < 48000; i++) {
      while (Serial.available() <= 0);
      byte msb = (byte)Serial.read();  // Most Significant Byte
      
      while (Serial.available() <= 0);
      byte lsb = (byte)Serial.read();  // Least Significant Byte

      // Reassemble the 16-bit signed integer from MSB and LSB
      int16_t sample = (msb << 8) | lsb;
      audio_buffer[index++] = sample;
    }

    // Wait for the end of transmission synchronization bytes
    while (Serial.available() <= 0);
    byte r_data_3 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte r_data_2 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte r_data_1 = (byte)Serial.read();

    // Check if the final synchronization bytes match
    if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
      // Send final acknowledgment bytes back to the laptop
      Serial.write(s_data_1);
      Serial.write(s_data_1);
      Serial.write(s_data_2);
      Serial.write(s_data_3);

      // Data has been successfully received
      return;
    }
  }

  // If anything goes wrong, stay in an infinite loop
  while (true);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void transfer_sensor_data_to_laptop() {

  byte s_data_1 = 0x01;
  byte s_data_2 = 0x02;
  byte s_data_3 = 0x03;

  // Write synchronization bytes to the laptop
  Serial.write(s_data_1);
  Serial.write(s_data_2);
  Serial.write(s_data_3);

  // Wait for acknowledgment from the laptop
  while (Serial.available() <= 0);
  byte r_data_3 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_2 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_1 = (byte)Serial.read();

  // Check if the acknowledgment matches expected values
  if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
    // Read temperature and humidity from DHT22 sensor
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    // Check if reading failed
    if (isnan(temperature) || isnan(humidity)) {
      temperature = 0.00f ;
      humidity = 0.00f ;
    }

    // Convert the float values to integers
    int temp_int = (int)temperature;  // Cast temperature to integer
    int hum_int = (int)humidity;      // Cast humidity to integer

    // Break down the temperature and humidity integers into bytes
    byte temp_msb = (temp_int >> 8) & 0xFF;  // Most significant byte of temperature
    byte temp_lsb = temp_int & 0xFF;         // Least significant byte of temperature
    byte hum_msb = (hum_int >> 8) & 0xFF;    // Most significant byte of humidity
    byte hum_lsb = hum_int & 0xFF;           // Least significant byte of humidity

    // Send the temperature and humidity bytes
    Serial.write(temp_msb);
    Serial.write(temp_lsb);
    Serial.write(hum_msb);
    Serial.write(hum_lsb);

    // Send the end of transfer signal
    Serial.write(s_data_1);
    Serial.write(s_data_2);
    Serial.write(s_data_3);

    return;
  }

  // If acknowledgment doesn't match, halt
  while (true);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void recieve_control_signal () {
  byte s_data_1 = 0x01;
  byte s_data_2 = 0x02;
  byte s_data_3 = 0x03;
  
  // Wait for initial synchronization bytes from the laptop
  while (Serial.available() <= 0);
  byte r_data_3 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_2 = (byte)Serial.read();
  while (Serial.available() <= 0);
  byte r_data_1 = (byte)Serial.read();

  // Check if synchronization bytes match
  if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
    // Send acknowledgment bytes back to the laptop
    Serial.write(s_data_1);
    Serial.write(s_data_1);
    Serial.write(s_data_2);
    Serial.write(s_data_3);

    // Actual signal data
    while (Serial.available() <= 0);
    byte x_data_1 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte x_data_2 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte x_data_3 = (byte)Serial.read();

    bool control_flow = false ;
    if (x_data_1 == 0x01 && x_data_2 == 0x02 && x_data_3 == 0x03) { 
      control_flow = true ;
    }

    // Wait for the end of transmission synchronization bytes
    while (Serial.available() <= 0);
    byte r_data_3 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte r_data_2 = (byte)Serial.read();
    while (Serial.available() <= 0);
    byte r_data_1 = (byte)Serial.read();

    // Check if the final synchronization bytes match
    if (r_data_3 == 0x03 && r_data_2 == 0x02 && r_data_1 == 0x01) {
      // Send final acknowledgment bytes back to the laptop
      Serial.write(s_data_1);
      Serial.write(s_data_1);
      Serial.write(s_data_2);
      Serial.write(s_data_3);

      // Calling Requested Function
      if (control_flow) transfer_sensor_data_to_laptop() ;

      // Data has been successfully received
      return;
    }
  }

  // If anything goes wrong, stay in an infinite loop
  while (true);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------


void setup() {

  dht.begin();
  setupI2S_speaker() ;
  setupI2S_microphone() ;
  setup_serial_communication () ;

  connect_to_laptop () ;
  delay(1000) ;
  record_audio() ;
  transfer_data_to_laptop() ;
  delay(1000) ;
  recieve_control_signal() ;
  recieve_data_from_laptop() ;
  delay(1000) ;
  play_audio() ;
  delay(1000) ;

}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void loop() {

}
