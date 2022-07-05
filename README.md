# MoCapRasps ![Status](https://img.shields.io/static/v1?style=flat&logo=github&label=status&message=finished&color=red) [![GitHub license](https://img.shields.io/github/license/debOliveira/MoCapRasps.svg)](https://github.com/debOliveira/MoCapRasps/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)  [![GitHub forks](https://img.shields.io/github/forks/debOliveira/MoCapRasps.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/debOliveira/MoCapRasps/network/) [![GitHub stars](https://img.shields.io/github/stars/debOliveira/MoCapRasps.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/debOliveira/MoCapRaspsn/stargazers/) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FdebOliveira%2FMoCapRasps&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

This repository stores an **open optical tracking/motion capture system using passive reflective markers and Raspberry Pi** boards as capturing station. This work is an result of my master dissertation entitled "Heterogeneous optical tracking system for mobile vehicles". 

## Materials

- Two (*minimum*) Raspberry 3b or above
- Two (*minimum*) NoIR cameras :arrow_right: **V2 cameras are preferable** if it's required more than 90FPS
- Long Ethernet cables that cover from the router/switcher to each RPi
- High support, like tripods or shelfs (do not forget to get some [gymbals](https://www.amazon.com.br/gp/product/B099HPMZK1/ref=ppx_yo_dt_b_asin_title_o02_s01?ie=UTF8&psc=1))
- Coolers, one for each RPi (I disassembled coolers from old PCS and used the ventilator part over the RPi)
- IR LED [rings](https://produto.mercadolivre.com.br/MLB-2096109150-led-infravermelho-cameras-seguranca-com-sensor-kit-4-placa-_JM) or [reflectors](https://produto.mercadolivre.com.br/MLB-705743885-refletor-72-leds-infravermelho-para-camera-de-seguranca-_JM#position=18&search_layout=stack&type=item&tracking_id=f82f63b5-6055-4f00-a978-0f2bfc703d91) (the rings give a better brigthness because they are easier to align with the camera axis)
- IR low-pass filters (using floppy disks or [glass](https://pt.aliexpress.com/item/1005003709944263.html?spm=a2g0o.order_list.0.0.1856caa4oP6TAy&gatewayAdapt=glo2bra))
- Polystyrene balls (I used 20mm ones)
- [Reflective adehesive](https://dmrefletivos.com.br/sinalizacao-viaria/grau-comercial/)
- Hard cardboard to construct the camera boxes.
- NPN FET, optocoupler, 3x 1kΩ resistors for each turn on/off circuit (you can also weld the power direct to the led ring and manually turn it on or off)

## Arena pictures

- [Full arena with 2 cameras](https://photos.app.goo.gl/bub5wwFEi5pR14ov5)
- Capture station [[1](https://photos.app.goo.gl/WmQuUtYHV3ZL1rLY8)] and [[2](https://photos.app.goo.gl/WmQuUtYHV3ZL1rLY8)]
- Camera box [[1](https://photos.app.goo.gl/25dToDRJKosVdfQk9)], [[2](https://photos.app.goo.gl/nADaQRypZxPyUfSX9)] and [[3](https://photos.app.goo.gl/5oBvEkRgRzf9mKXy9)]

## Construction

### Camera

1) Connect the camera to a RPi and capture a calibration data set using [`./calib`](/calib/). Then run [camera calibrator application](https://github.com/debOliveira/myCameraCalibrator).
1) Construct the boxes to the cameras using cardboard. I used a 8cm X 8cm X 4cm box. Remeber to cut a hole in the center for the camera lens and the LED power input, and at the back for the flat cable and power cable of the LED ring. Leave the front open.
1) I cutted a 3mm X 3mm square at the bottom to fit the gymbal tip. Then I did four little holes to pass a string (in a X format, at the outter side of the boox) to hold the box to the gymbal base.
1) Weld the LED to the power source or connect it to the circuit pins. You may find the design of the circuit board here.
1) Put the filter over the camera lens and cover the rest with 
insulating tape. I did the hoel for the lens using a compass. 
1) Put the camera+filter back at the box, check if the LED is powering on and fit the gymbal tip through the hole at the bottom of the box. Then pass the string behind the gymbal bar and tie a not to secure it over the gymbal base.
1) If everything is working, you can now close de box. Double check to avoid opening the box multiple times
1) If you have light leakeage trough the lens, add a black cardboard ring inside the LED ring. 

### Server

1) Read the instructions at [`./central`](/central/)

### Clients

1) Connect each RPi to a camera. The system has a 1cm accuracy when using 4 cameras, but the number can be expanded to improve cover area
1) Put a cooler over each RPi. The high FPS will heat to 80C and activate frequency capping if no ventilation is directly over the processor
1) Read the instructions at [`./collect`](/collect/)

## Usage

Each folder has its own `README.md` guide file. Please read each one and adjust both your server and clients. 

## Organisation

- [`./calib`](/calib/) stores the code to collect the images used in intrinsics calibration. 
- [`./central`](/central/) has the codes of the server side.
- [`./collect`](/collect/) comprises the files used in the clients. 
