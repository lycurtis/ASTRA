/*
 * i2c.h
 *
 *  Created on: Jun 23, 2025
 *      Author: cly
 */

//#ifndef I2C_H_
//#define I2C_H_

#pragma once // error prone alternative to ifndef...

#include <stdint.h>

// I2C API
void I2C_Init(void);

void I2C_Write(uint8_t dev_addr, uint8_t *data, uint16_t len);
void I2C_Read(uint8_t dev_addr, uint8_t *data, uint16_t len);

void I2C_Mem_Write(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len);
void I2C_Mem_Read(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len);



//#endif /* I2C_H_ */
