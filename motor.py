from gpiozero import PWMOutputDevice
from time import sleep

ena = PWMOutputDevice(18)  # change to your ENA pin BCM
enb = PWMOutputDevice(13)  # change to your ENB pin BCM

while True:
	print("Testing ENA pin PWM")
	for duty in [0, 0.25, 0.5, 0.75, 1.0]:
		ena.value = duty
		print(f"ENA duty cycle: {duty}")
		sleep(2)

	ena.value = 0
	print("Testing ENB pin PWM")
	for duty in [0, 0.25, 0.5, 0.75, 1.0]:
		enb.value = duty
		print(f"ENB duty cycle: {duty}")
		sleep(2)

	enb.value = 0
	print("Done")
