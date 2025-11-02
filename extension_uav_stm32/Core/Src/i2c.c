/*
 * i2c.c
 *
 *  Created on: Jun 23, 2025
 *      Author: cly
 */
#include "i2c.h"

// Include your STM32 register headers here
#include "stm32f4xx.h"

/*
 * SCL: PB8
 * SDA: PB9
 * */
#define AC_GPIOB_OTYPER_OT8 (1U<<8)
#define AC_GPIOB_OTYPER_OT9 (1U<<9)
#define I2C_100KHZ 80
#define SD_MODE_MAX_RISE_TIME 17
#define I2C1_AF4 (4U)

void I2C_Init(void) {
	// Enable clock access for GPIOB
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;

	// Configure GPIO pins for I2C (Alternate Function, Open Drain)
	// Set  PB8 and PB9 mode to alternate function [10] for I2C1
	GPIOB->MODER |= (1U<<17);
	GPIOB->MODER &= ~(1U<<16);

	GPIOB->MODER |= (1U<<19);
	GPIOB->MODER &= ~(1U<<18);

	// Configure Open-Drain output type
	GPIOB->OTYPER |= AC_GPIOB_OTYPER_OT8; // Output open drain pin 8
	GPIOB->OTYPER |= AC_GPIOB_OTYPER_OT9; // Output open drain pin 9

	// Enable Pull-up [01] for PB8 and PB9
	GPIOB->PUPDR &= ~(1U<<17);
	GPIOB->PUPDR |= (1U<<16);

	GPIOB->PUPDR &= ~(1U<<19);
	GPIOB->PUPDR |= (1U<<18);

	// Configure Alternate Function I2C1 (AF4) for PB8 and PB9
	// We know AF4 based on Alternate function mapping on datasheet
	GPIOB->AFR[1] |= (4U << 0);  // PB8: AFRH[3:0]
	GPIOB->AFR[1] |= (4U << 4);  // PB9: AFRH[7:4]


    // Enable clock access to I2C1 peripheral
	RCC->APB1ENR |= RCC_APB1ENR_I2C1EN;

    // RESET I2C1
	// 1. Enter RESET Mode then immediately come out of RESET
	I2C1->CR1 |= I2C_CR1_SWRST;
	I2C1->CR1 &= ~I2C_CR1_SWRST;

	// 2. Set clock Frequency to 16 MHz
	I2C1->CR2 = 16U; //I2C1->CR2 |= (1U<<4); //FREQ[5:0] ==> (1U<<4) = 0b0100 = 16


	// 3. Set I2C to standard mode (100 kHz)
	/*
	 * CCR = (Peripheral Clock) / (2 * I2C Speed)
	 * CCR = 16 MHz / (2 * 100 kHz) = 80
	 *
	 * TRISE = (Maximum rise time/ T_PCLK1) + 1
	 * TRISE = (1000 ns / 62.5 ns) + 1
	 * TRISE = 16 + 1 = 17
	 * */
	I2C1->CCR = I2C_100KHZ;

	// 4. Set rise time
	I2C1->TRISE = SD_MODE_MAX_RISE_TIME;

	// 5. Enable I2C peripheral
	I2C1->CR1 |= I2C_CR1_PE;
}

void I2C_Write(uint8_t dev_addr, uint8_t *data, uint16_t len) {
    // Implement I2C write sequence (START, SEND ADDR, WRITE DATA, STOP)

	// 1. Generate a START condition (set the START bit in CR1)
	I2C1->CR1 |= I2C_CR1_START;

	// 2. Wait until the SB (Start Bit) flag is set in SR1 (indicates start has been generated)
	while(!(I2C1->SR1 & I2C_SR1_SB)){}

	// 3. Send the slave address with the write bit (LSB = 0)
	I2C1->DR = dev_addr << 1;

	// 4. Wait until the ADDR (Address Sent) flag is set in SR1
	while(!(I2C1->SR1 & I2C_SR1_ADDR)){}

	// 5. Clear the ADDR flag by reading SR1 followed by SR2 (this is required)
	(void)I2C1->SR1;
	(void)I2C1->SR2;

	// 6. Loop through the data array:
	//    For each byte:
	//    a. Write the byte to the DR (Data Register)
	//    b. Wait until the TXE (Transmit Data Register Empty) flag is set (means you can send the next byte)
	for(int i = 0; i < len; i++){
		I2C1->DR = data[i];
		while(!(I2C1->SR1 & I2C_SR1_TXE)){}
	}

	// 7. After all bytes are sent, wait until BTF (Byte Transfer Finished) flag is set (optional but recommended)
	while(!(I2C1->SR1 & I2C_SR1_BTF)){}
	// 8. Generate a STOP condition (set the STOP bit in CR1)
	I2C1->CR1 |= I2C_CR1_STOP;
}

void I2C_Read(uint8_t dev_addr, uint8_t *data, uint16_t len) {
    // Implement I2C read sequence (START, SEND ADDR, READ DATA, STOP)

	// 1. Enable ACK generation
	    I2C1->CR1 |= I2C_CR1_ACK;

	    // 2. Generate a START condition
		I2C1->CR1 |= I2C_CR1_START;
		while(!(I2C1->SR1 & I2C_SR1_SB));

	    // 3. Send the slave address with the READ bit (LSB = 1)
	    // FIX: Address must be shifted and OR'd with 1 for read.
		I2C1->DR = (dev_addr << 1) | 1;
		while(!(I2C1->SR1 & I2C_SR1_ADDR));

	    // 4. Clear the ADDR flag
		(void)I2C1->SR1;
		(void)I2C1->SR2;

	    // 5. Loop to read bytes
	    for (int i = 0; i < len; i++) {
	        if (i == len - 1) {
	            // For the last byte, we need to NACK it and issue a STOP condition.
	            // a. Disable ACK
	            I2C1->CR1 &= ~I2C_CR1_ACK;
	            // b. Generate STOP
	            I2C1->CR1 |= I2C_CR1_STOP;
	        }

	        // Wait until RXNE (Receive Buffer Not Empty) flag is set
	        while(!(I2C1->SR1 & I2C_SR1_RXNE));

	        // Read data from DR
	        data[i] = I2C1->DR;
	    }

	    // 6. Re-enable ACK for future transfers
	    I2C1->CR1 |= I2C_CR1_ACK;
}


void I2C_Mem_Write(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len) {
    // Send register address, then write data

	// 1. Generate a START condition (set the START bit in CR1)
	I2C1->CR1 |= I2C_CR1_START;

	// 2. Wait until the SB (Start Bit) flag is set in SR1
	while(!(I2C1->SR1 & I2C_SR1_SB)){}

	// 3. Send the slave address with the write bit (LSB = 0)
	I2C1->DR = dev_addr << 1;

	// 4. Wait until the ADDR (Address Sent) flag is set in SR1
	while(!(I2C1->SR1 & I2C_SR1_ADDR)){}

	// 5. Clear the ADDR flag by reading SR1 followed by SR2
	// To clear this flag (which is required to continue the transfer):
	(void)I2C1->SR1; // Read SR1
	(void)I2C1->SR2; // Then immediately after read SR2

	// 6. Send the memory/register address (mem_addr)
	while(!(I2C1->SR1 & I2C_SR1_TXE)); // Must wait for TXE before sending register address
	I2C1->DR = reg_addr;

//	// 7. Wait until the TXE (Transmit Data Register Empty) flag is set
//	while(!(I2C1->SR1 & I2C_SR1_TXE)){}

	// 8. Loop through the data array:
	//    For each byte:
	//    a. Wait until TXE is set
	//    b. Write the byte to the DR register
	for(int i = 0; i < len; i++){

		while(!(I2C1->SR1 & I2C_SR1_TXE)){}

		I2C1->DR = data[i]; // Write the current byte to the DR register
	}

	// 9. After all bytes are sent, wait until BTF (Byte Transfer Finished) flag is set (optional but recommended)
	while(!(I2C1->SR1 & I2C_SR1_TXE)){}
	while(!(I2C1->SR1 & I2C_SR1_BTF)){}

	// 10. Generate a STOP condition (set the STOP bit in CR1)
	I2C1->CR1 |= I2C_CR1_STOP;
}

void I2C_Mem_Read(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, uint16_t len) {
    // Send register address, then read data
	/*
	 * [START] → [Slave Address + Write] → [Register Address] →
	 * → [REPEATED START] → [Slave Address + Read] → [Read Data] → [STOP]
	 * */
	/*
	 * When you use Mem_Read (Memory/Register Read) on devices like the MPU6050,
	 * you need to:
		Tell the device which internal memory/register you want to read from.
		Then actually read the data from that register.
	 */
	// Send register address, then read data (Combined Transaction)

		// ====== Phase 1: Write the register address to read from ======

		// 1. Generate a START condition
		I2C1->CR1 |= I2C_CR1_START;
		while(!(I2C1->SR1 & I2C_SR1_SB));

		// 2. Send the slave address with the WRITE bit
		I2C1->DR = dev_addr << 1;
		while(!(I2C1->SR1 & I2C_SR1_ADDR));

		// 3. Clear the ADDR flag
		(void)I2C1->SR1;
		(void)I2C1->SR2;

		// 4. Send the memory/register address
		while(!(I2C1->SR1 & I2C_SR1_TXE));
		I2C1->DR = reg_addr;

		// 5. Wait for TXE flag to set, indicating address is sent
		while(!(I2C1->SR1 & I2C_SR1_TXE));


		// ====== Phase 2: Read data from the specified register ======

		// 6. Generate a REPEATED START condition
		I2C1->CR1 |= I2C_CR1_START;
		while(!(I2C1->SR1 & I2C_SR1_SB));

		// 7. Send the slave address with the READ bit
		I2C1->DR = (dev_addr << 1) | 1;
		while(!(I2C1->SR1 & I2C_SR1_ADDR));

		// 8. Clear the ADDR flag
		(void)I2C1->SR1;
		(void)I2C1->SR2;

		// 9. Read the data bytes (using the same logic as I2C_Read)
	    // FIX: Use the corrected read logic here as well.
	    for (int i = 0; i < len; i++) {
	        if (i == len - 1) {
	            // For the last byte, NACK it and STOP.
	            I2C1->CR1 &= ~I2C_CR1_ACK;
	            I2C1->CR1 |= I2C_CR1_STOP;
	        }

	        while(!(I2C1->SR1 & I2C_SR1_RXNE));
	        data[i] = I2C1->DR;
	    }

	    // 10. Re-enable ACK for future transfers
	    I2C1->CR1 |= I2C_CR1_ACK;
}
