#include "globals.h"
#include <stddef.h>
#include "FreeRTOS.h"
#include "semphr.h"

// IMU data
comp imu_CA = {0};
float roll  = 0.0f;
float pitch = 0.0f;

// RC Channel data
volatile uint16_t ibus_channels[6] = {0};

// iBUS DMA buffer and semaphore
uint8_t ibus_dma_buffer[IBUS_PACKET_SIZE];
SemaphoreHandle_t ibus_rx_semaphore = NULL;
