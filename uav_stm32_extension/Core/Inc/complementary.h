#pragma once


typedef struct{
	float roll;
	float pitch;
} comp;

void ComplementaryFilter_Init(comp *filter);
void ComplementaryFilter_Update(comp *filter, float ax, float ay, float az, float gx, float gy, float dt);
