/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "i2c.h"
#include "mpu6050.h"
#include "complementary.h"
#include "uart.h"      // My personal with IBUS_ReadByte()
#include <stdio.h>
#include "globals.h"
#include "semphr.h"
#include "usart.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef uint32_t TaskProfiler;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */
// Task Handles
TaskHandle_t IMU_TaskHandle = NULL;
TaskHandle_t IBUS_TaskHandle = NULL;
// Task Profilers
TaskProfiler IMUTaskProfiler;
TaskProfiler IBUSTaskProfiler;
/* USER CODE END Variables */
/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */
void IMU_Task(void *pvParameters);
void IBUS_Task(void *pvParameters);
/* USER CODE END FunctionPrototypes */

void StartDefaultTask(void *argument);

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
	ibus_rx_semaphore = xSemaphoreCreateBinary();
	configASSERT(ibus_rx_semaphore != NULL);
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  xTaskCreate(IMU_Task,  "IMU",  256, NULL, tskIDLE_PRIORITY+2, &IMU_TaskHandle);
  xTaskCreate(IBUS_Task, "IBUS", 256, NULL, tskIDLE_PRIORITY+1, &IBUS_TaskHandle);
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

}

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* USER CODE BEGIN StartDefaultTask */
  /* Infinite loop */
  for(;;)
  {
    osDelay(1);
  }
  /* USER CODE END StartDefaultTask */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */
void IMU_Task(void *pvParameters){
	const TickType_t period = pdMS_TO_TICKS(5); // 200 Hz
	TickType_t last = xTaskGetTickCount();

	while(1){
		float ax_g, ay_g, az_g;
		float gx_dps, gy_dps, gz_dps;

		MPU6050_Read_Calibrated(&ax_g, &ay_g, &az_g, &gx_dps, &gy_dps, &gz_dps);
		ComplementaryFilter_Update(&imu_CA, ax_g, ay_g, az_g, gx_dps, gy_dps, 0.005f);

		// results
		roll = imu_CA.roll;
		pitch = imu_CA.pitch;

		// Optional print statements here for roll pitch
		//char buffer[64];
		//snprintf(buffer, sizeof(buffer), "Roll: %.2f\tPitch: %.2f\r\n", roll, pitch);
		//UART2_WriteString(buffer);
		IMUTaskProfiler++;

		vTaskDelayUntil(&last, period);
	}
}

void IBUS_Task(void *pvParameters){
	__HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);
	HAL_UART_Receive_DMA(&huart1, ibus_dma_buffer, IBUS_PACKET_SIZE);

	while(1){
		// Wait indefinitely for the semaphore, the task will be in blocked state
		// ISR is in charge of giving the semaphore
		if(xSemaphoreTake(ibus_rx_semaphore, portMAX_DELAY) == pdTRUE){
			if(ibus_dma_buffer[0] == 0x20 && ibus_dma_buffer[1] == 0x20){ //ibus_dma_buffer[1] == 0x40
				// Decode channels Note: iBUS data is little endian
				for(int ch = 0; ch < 6; ch++){
					ibus_channels[ch] = (uint16_t)(ibus_dma_buffer[2 + ch*2] | (ibus_dma_buffer[3 + ch*2] << 8));
					//ibus_channels[ch] = (uint16_t)(ibus_dma_buffer[ch*2 + 3] << 8) | ibus_dma_buffer[ch*2 + 2];
				}

//				char str[64];
//				snprintf(str, sizeof(str), "CH1:%4d CH2:%4d CH3:%4d CH4:%4d\r\n",
//						ibus_channels[0], ibus_channels[1],
//						ibus_channels[2], ibus_channels[3]);
//				UART2_WriteString(str);
			}
			IBUSTaskProfiler++;
		}
	}
}
/* USER CODE END Application */

