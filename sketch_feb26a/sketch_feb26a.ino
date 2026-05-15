#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu(0x68);

// 🔹 Offsets
float gx_offset = 0, gy_offset = 0, gz_offset = 0;

// 🔹 Angles
float pitch = 0, roll = 0;

// 🔹 Smoothed gyro
float gx_s = 0, gy_s = 0;
float alpha = 0.2;

// 🔹 Timing
unsigned long prevTime = 0;

void calibrateGyro() {
  Serial.println("Calibrating... keep still");

  for (int i = 0; i < 1000; i++) {
    int16_t gx, gy, gz;
    mpu.getRotation(&gx, &gy, &gz);

    gx_offset += gx;
    gy_offset += gy;
    gz_offset += gz;

    delay(2);
  }

  gx_offset /= 1000;
  gy_offset /= 1000;
  gz_offset /= 1000;

  Serial.println("Done calibration");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);

  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("MPU6050 OK");
  } else {
    Serial.println("MPU6050 check failed (still usable)");
  }

  calibrateGyro();
  prevTime = millis();
}

void loop() {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  // 🔹 Time
  unsigned long now = millis();
  float dt = (now - prevTime) / 1000.0;
  prevTime = now;

  // 🔹 Read sensor
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // 🔹 Apply offset
  gx -= gx_offset;
  gy -= gy_offset;
  gz -= gz_offset;

  // 🔹 Smooth gyro
  gx_s = alpha * gx + (1 - alpha) * gx_s;
  gy_s = alpha * gy + (1 - alpha) * gy_s;

  float gx_rate = gx_s / 131.0;
  float gy_rate = gy_s / 131.0;

  // 🔹 SAFE accel calculations
  float accel_pitch = 0;
  float accel_roll = 0;

  float denom = sqrt((float)ay * ay + (float)az * az);

  if (denom > 0.001) { // 🔥 prevent division issues
    accel_roll = atan2(-ax, denom) * 180 / PI;
  }

  accel_pitch = atan2((float)ay, (float)az) * 180 / PI;

  // 🔥 Complementary filter
  pitch = 0.98 * (pitch + gx_rate * dt) + 0.02 * accel_pitch;
  roll  = 0.98 * (roll  + gy_rate * dt) + 0.02 * accel_roll;

  Serial.print("{\"ax\":");
  Serial.print(ax);
  Serial.print(",\"ay\":");
  Serial.print(ay);
  Serial.print(",\"az\":");
  Serial.print(az);
  Serial.print(",\"gx\":");
  Serial.print(gx);
  Serial.print(",\"gy\":");
  Serial.print(gy);
  Serial.print(",\"gz\":");
  Serial.print(gz);
  Serial.println("}");

  delay(1000);
}