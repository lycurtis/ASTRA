/*
 * complementary.c
 *
 *  Created on: Jul 13, 2025
 *      Author: cly
 */


#include "complementary.h"
#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#define ALPHA 0.98f

void ComplementaryFilter_Init(comp *filter){
	filter->roll = 0.0f;
	filter->pitch = 0.0f;
}

void ComplementaryFilter_Update(comp *filter, float ax, float ay, float az, float gx, float gy, float dt){
	// Calculate roll and pitch from the accelerometer
	float roll_accel = atan2f(ay, az) * (180.0f/PI);
	float pitch_accel = atan2f(-ax, sqrtf(ay * ay + az * az)) * (180.0f/PI);

	// Complementary filter formula
	// angle = α * (angle + gyro * dt) + (1 - α) * accel_angle;
	filter->roll = ALPHA * (filter->roll+gx*dt) + (1.0f-ALPHA) * roll_accel;
	filter->pitch = ALPHA * (filter->pitch+gy*dt) + (1.0f-ALPHA) * pitch_accel;

}
