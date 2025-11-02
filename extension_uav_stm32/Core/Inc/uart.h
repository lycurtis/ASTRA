/*
 * uart.h
 *
 *  Created on: Jul 16, 2025
 *      Author: cly
 */

//#ifndef UART_H_
//#define UART_H_
//
//
//
//#endif /* UART_H_ */

#pragma once

#include <stdint.h>
#include <stdio.h>

// UART 1
void UART1_Init(uint32_t baudrate);
uint8_t IBUS_ReadByte(void);

// UART 2
void UART2_Init(uint32_t baudrate);
void UART2_WriteChar(char c);
void UART2_WriteString(const char *str);
char UART2_ReadChar(void);
