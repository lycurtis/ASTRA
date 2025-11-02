
#include "mpu6050.h"
#include "i2c.h"


//void MPU6050_Init(void){
//	uint8_t data = 0x00; // Wake up device
//	I2C_Mem_Write(MPU6050_I2C_ADDR, MPUR_PWR_MGMT_1, &data, 1);
//}


// Calibration offsets
static int16_t accel_x_offset = 0, accel_y_offset = 0, accel_z_offset = 0;
static int16_t gyro_x_offset = 0, gyro_y_offset = 0, gyro_z_offset = 0;

uint8_t MPU6050_Init(void){
	uint8_t check;
	uint8_t data;

	//Initialize the I2C peripheral
	//I2C_Init(); //Optional placement here or place in main file

	//Check the WHO_AM_I register to confirm communication with the MPU6050
	//The MPU6050's I2C address is 0x68, therefore this register should also contain 0x68
	I2C_Mem_Read(MPU6050_I2C_ADDR, MPU6050_WHO_AM_I, &check, 1);

	if(check == MPU6050_I2C_ADDR){
		//Must wake up device by writing 0 to the Power Management 1 Register
		data = 0x00;
		I2C_Mem_Write(MPU6050_I2C_ADDR, MPUR_PWR_MGMT_1, &data, 1);

		// Set the sample rate. 1kHz/(1+SMPLRT_DIV)
		// Set SMPLRT_DIV to 7 for a 125Hz sample rate
		data = 0x07;
		I2C_Mem_Write(MPU6050_I2C_ADDR, MPUR_SMPLRT_DIV, &data, 1);

		// Configure the accelerometer to a full-scale range of +/- 2g
		data = 0x00;
		I2C_Mem_Write(MPU6050_I2C_ADDR, MPUR_ACCEL_CONFIG, &data, 1);

		// Configure the gyroscope to a full-scale range of +/- 250 degrees/sec
		data = 0x00;
		I2C_Mem_Write(MPU6050_I2C_ADDR, MPUR_GYRO_CONFIG, &data, 1);

		return 1; // success
	}

	return 0; // fail
}

void MPU6050_Read_Accel(int16_t *accel_x, int16_t * accel_y, int16_t *accel_z){
	uint8_t buffer[6]; //stores raw data

	// Read 6 bytes of data starting from the ACCEL_XOUT_H register
	// Note: The MPU will automatically increment the register address after each read
	I2C_Mem_Read(MPU6050_I2C_ADDR, MPUR_ACCEL_XOUT_H, buffer, 6);

	// data is transmitted in big-endian format (MSB first)
	// we combine the high and low bytes for each axis to form a 16-bit signed integer
	*accel_x = (int16_t)(buffer[0] << 8 | buffer[1]);
	*accel_y = (int16_t)(buffer[2] << 8 | buffer[3]);
	*accel_z = (int16_t)(buffer[4] << 8 | buffer[5]);
}

void MPU6050_Read_Gyro(int16_t *gyro_x, int16_t *gyro_y, int16_t *gyro_z){
	uint8_t buffer[6];

	I2C_Mem_Read(MPU6050_I2C_ADDR, MPUR_GYRO_XOUT_H, buffer, 6);

	*gyro_x = (int16_t)(buffer[0] << 8 | buffer[1]);
	*gyro_y = (int16_t)(buffer[2] << 8 | buffer[3]);
	*gyro_z = (int16_t)(buffer[4] << 8 | buffer[5]);
}


void MPU6050_Calibrate(void){
	// Calibrate accel
	int32_t sum_x = 0, sum_y = 0, sum_z = 0;
	const int samples = 500;
	for (int i = 0; i < samples; i++) {
		int16_t ax, ay, az;
		MPU6050_Read_Accel(&ax, &ay, &az);
		sum_x += ax;
		sum_y += ay;
		sum_z += az;
	}
	accel_x_offset = sum_x / samples;
	accel_y_offset = sum_y / samples;
	accel_z_offset = (sum_z / samples) - 16384; // Optional: subtract 16384 if you want "0g at rest"

	// Calibrate gyro
	sum_x = sum_y = sum_z = 0;
	for (int i = 0; i < samples; i++) {
		int16_t gx, gy, gz;
		MPU6050_Read_Gyro(&gx, &gy, &gz);
		sum_x += gx;
		sum_y += gy;
		sum_z += gz;
	}
	gyro_x_offset = sum_x / samples;
	gyro_y_offset = sum_y / samples;
	gyro_z_offset = sum_z / samples;
}


void MPU6050_Read_Calibrated(float *ax_g_out, float *ay_g_out, float *az_g_out, float *gx_dps_out, float *gy_dps_out, float *gz_dps_out){
	 int16_t accel_x, accel_y, accel_z;
	    int16_t gyro_x, gyro_y, gyro_z;

	    MPU6050_Read_Accel(&accel_x, &accel_y, &accel_z);
	    MPU6050_Read_Gyro(&gyro_x, &gyro_y, &gyro_z);

	    // Offset correction
	    int16_t axc = accel_x - accel_x_offset;
	    int16_t ayc = accel_y - accel_y_offset;
	    int16_t azc = accel_z - accel_z_offset;

	    int16_t gxc = gyro_x - gyro_x_offset;
	    int16_t gyc = gyro_y - gyro_y_offset;
	    int16_t gzc = gyro_z - gyro_z_offset;

	    // Normalization
	    *ax_g_out = axc / 16384.0f;
	    *ay_g_out = ayc / 16384.0f;
	    *az_g_out = azc / 16384.0f;

	    *gx_dps_out = gxc / 131.0f;
	    *gy_dps_out = gyc / 131.0f;
	    *gz_dps_out = gzc / 131.0f;
}
