# CMPE264project1- HDR

## Information
**OS**: Mac/Windows
**Language**: Python 2.7.15
**Libraries Used**: OpenCV, Numpy, MatPlotLib, Scripy

## Description
Collaborative project (Joshua Pena & Eduardo Hirata) to develop manual HDR Photography composition. Program requires calibration images as well as 3 photographs at different exposures to make HDR composition.

## To Run Program:
```bash
python hdr_process.py
```

Keep the directories as such:
```bash
.
├── README.md
├── CMPE264_Project1_EH_JP.pdf
├── calibrationPhotos
│   ├── p
│   │   ├── p1.JPG
│   │   ├── p2.JPG
│   │   ├── p3.JPG
│   │   ├── p4.JPG
│   │   └── p5.JPG
├── device_values.txt
├── hdr_process.py
├── images
│   ├── 200.jpg
│   ├── 4000.jpg
│   └── 800.jpg
└── results
    ├── part_four
    ├── part_one
    ├── part_three
    └── part_two
```

## To run on a different set of images:
Update the values for the radiometric:
* Update the time values on line 207
* Update the path name on line 81

Update values for the hdr pictures:
* Update the values on line 13 to match the new time values.
* Update the path name on line 264
