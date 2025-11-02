/*
 * mpu6050.h
 *
 *  Created on: Jun 25, 2025
 *      Author: cly
 */

//#ifndef MPU6050_H_
//#define MPU6050_H_
//
//
//
//#endif /* MPU6050_H_ */

#pragma once

#include <stdint.h>

// MPU6050 I2C address
#define MPU6050_I2C_ADDR 0x68 // I2C address when AD0 pin is low
#define MPU6050_WHO_AM_I 0x75

// Accelerometer Measurements Register Map
#define MPUR_ACCEL_XOUT_H 0x3B // (X-axis high byte)
//#define MPUR_ACCEL_XOUT_L 0x3C // (X-axis low byte)
#define MPUR_ACCEL_YOUT_H 0x3D
//#define MPUR_ACCEL_YOUT_L 0x3E
#define MPUR_ACCEL_ZOUT_H 0x3F
//#define MPUR_ACEEL_ZOUT_L 0x40


// Gyroscope Measurements Register Map
#define MPUR_GYRO_XOUT_H 0x43
//#define MPUR_GYRO_XOUT_L 0x44
#define MPUR_GYRO_YOUT_H 0x45
//#define MPUR_GYRO_YOUT_L 0x46
#define MPUR_GYRO_ZOUT_H 0x47
//#define MPUR_GYRO_ZOUT_L 0x48

// Misc.
#define MPUR_SMPLRT_DIV 0x19 // Sample Rate Divider
#define MPUR_PWR_MGMT_1 0x6B // Power Management 1
#define MPUR_ACCEL_CONFIG 0x1C
#define MPUR_GYRO_CONFIG 0x1B

//void MPU6050_Init(void);
uint8_t MPU6050_Init(void);
void MPU6050_Read_Accel(int16_t *accel_x, int16_t * accel_y, int16_t *accel_z);
void MPU6050_Read_Gyro(int16_t *gyro_x, int16_t *gyro_y, int16_t *gyro_z);

void MPU6050_Calibrate(void);
void MPU6050_Read_Calibrated(float *ax_g, float *ay_g, float *az_g, float *gx_dps, float *gy_dps, float *gz_dps);
