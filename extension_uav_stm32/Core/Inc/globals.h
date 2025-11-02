#pragma once
#include <stdint.h>
#include <stddef.h>        // for NULL
#include "complementary.h" // comp
#include "FreeRTOS.h"
#include "semphr.h"

#define IBUS_PACKET_SIZE 32

extern comp  imu_CA;
extern float roll;
extern float pitch;
extern volatile uint16_t ibus_channels[6];

extern uint8_t ibus_dma_buffer[IBUS_PACKET_SIZE];
extern SemaphoreHandle_t ibus_rx_semaphore;
