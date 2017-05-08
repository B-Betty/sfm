**This code converts image data and imu data from IOS devices to ros message.**

1. add raw image folder, image_time file, imu_data file at /data/

2. cd raw image folder, then

   $ ls | cat -n | while read n f; do mv "$f" "$n.PNG"; done 

   to rename all the image by time sequence

3. $ roslaunch sfm sfm.launch