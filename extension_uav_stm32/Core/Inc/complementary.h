///*
// * complementary.h
// *
// *  Created on: Jul 13, 2025
// *      Author: cly
// */
//
//#ifndef COMPLEMENTARY_H_
//#define COMPLEMENTARY_H_
//
//
//
//#endif /* COMPLEMENTARY_H_ */

#pragma once


typedef struct{
	float roll;
	float pitch;
} comp;

void ComplementaryFilter_Init(comp *filter);
void ComplementaryFilter_Update(comp *filter, float ax, float ay, float az, float gx, float gy, float dt);
