/*
 * uart.c
 *
 *  Created on: Jul 16, 2025
 *      Author: cly
 */
#include "uart.h"
#include "stm32f4xx.h"


// UART1 PA10
void UART1_Init(uint32_t baudrate){
	// 1. Enable clocks
	RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

	// 2. Set PA10 to Alternate Function (AF7)
	GPIOA->MODER &= ~(3 << (10 * 2));
	GPIOA->MODER |=  (2 << (10 * 2)); // Alternate Function mode
	GPIOA->AFR[1] &= ~(0xF << 8); // Clear bits for PA10
	GPIOA->AFR[1] |=  (7 << 8); // AF7 for USART1_RX

	// 3. Configure USART1
	USART1->BRR = 90000000 / 115200;     // 90 MHz / 115200 = ~781
	USART1->CR1 |= USART_CR1_RE;  // Enable receiver
	USART1->CR1 |= USART_CR1_UE; // Enable USART
}

uint8_t IBUS_ReadByte(void){
    while (!(USART1->SR & USART_SR_RXNE));
    return USART1->DR;
}

// UART 2
void UART2_Init(uint32_t baudrate) {
	// Configure UART GPIO pin (TX)
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN; // Enable clock access to GPIOA
	GPIOA->MODER |= (1U << 5);
	GPIOA->MODER &= ~(1U << 4); // Set PA2 to alternate function mode (MODER2)
	GPIOA->AFR[0] &= ~(1U << 11); // Set PA2 to USART2_TX (AF07 = 0111)
	GPIOA->AFR[0] |= (1U << 10); // note AFR[0] = AFRL(ow) vs AFR[1] = AFRH(igh)
	GPIOA->AFR[0] |= (1U << 9); // since PA2 is A pin 2 we want AFRL
	GPIOA->AFR[0] |= (1U << 8); // PORTA PIN2 is set to AF7 which corresponds to USART2_TX

	// Configure UART GPIO pin (RX)
	GPIOA->MODER |= (1U << 7);
	GPIOA->MODER &= ~(1U << 6); // Set PA3 mode to alternate function mode
	GPIOA->AFR[0] &= ~(1U << 15); // Set PA3 alternate function type to UART_RX (AF07)
	GPIOA->AFR[0] |= (1U << 14);
	GPIOA->AFR[0] |= (1U << 13);
	GPIOA->AFR[0] |= (1U << 12);

	// Configure UART module
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN; // Enable clock access to UART 2
	USART2->BRR = (45000000U / baudrate);// Configure baudrate USART2->BRR = SYS_FREQ / baudrate;
	USART2->CR1 = (USART_CR1_TE | USART_CR1_RE); // Configure the transfer direction (Set TE1 to enable Transmitter) and RE for Receiver
	// Note: there is no | operator because we want to clean everything in the UART Control Register 1 (CR1)
	// while also setting CR1 bit 3 to 1 to enable transmitter
	USART2->CR1 |= USART_CR1_UE; // Enable UART Module (UE: USART enable)
	// Note: this requires an | operation because we also want to add on the UE bit
	// without the | operation it would set every bit to 0 besides the UE bit
	// Recall: Register = 0100 0000 vs Register |= 0100 0000 (is equivalent to a "+" operation)

	// Enable TX, RX, USART2
    //USART2->CR1 |= USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

void UART2_WriteChar(char c) {
    while (!(USART2->SR & USART_SR_TXE)) {}  // Wait until transmit buffer empty
    USART2->DR = (c & 0xFF);
}

void UART2_WriteString(const char *str) {
    while (*str) {
        UART2_WriteChar(*str++);
    }
}

char UART2_ReadChar(void) {
    while (!(USART2->SR & USART_SR_RXNE)) {}  // Wait until receive buffer not empty
    return (char)(USART2->DR & 0xFF);
}
