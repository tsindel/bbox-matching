# Bounding Box Matching
Bounding Box Matching using YOLOv4, implemented in Tensorflow and LabVIEW

This is an auxiliary repository to the "Bounding Box Matching: A Sparse Object-centric
Correspondence Method for Stereo Vision" paper.

To recreate the results in Fig. 8 of the paper, perform the following steps:
1) Clone the repo
2) Read README_yolov4 file in yolov4 subfolder and follow the steps to configure Tensorflow implementation of YOLOv4
3) Install LabVIEW Runtime 2019 or newer with Vision Development Module
4) Attach a stereo rig comprised of two cameras to the development computer
5) Calibrate the stereo rig using the VI in "calibrate" subfolder and save the calibration file. Replace the old file in "stereo" folder with the new one.
6) Open "main" VI in LabVIEW Runtime
7) Configure camera VIs to correspond to the connected stereo rig
8) Run the VI and observe the results

# Stereo rig geometry
To recreate the results in Fig. 5-7 of the paper, perform the following steps:
1) Open folder "stereo-rig-top" in separate project (IDE) and configure the interpreter (GPU enabled works better)
2) Import dependencies
3) If desired, modify param values in lines 188-189, 192-195, 204 and 221
4) Run the "main" script
