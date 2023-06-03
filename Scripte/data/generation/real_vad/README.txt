The ground-truth information for the RealVAD dataset:

The video can be downloaded from www.youtube.com/watch?v=51pRTOIso4U

We do not supply the raw images, but instead we supply the bounding boxes belong to each panelist (from 1 to 9) in a frame and the corresponding VAD ground-truth.

These are organized by the panelist identifier number into the files called Panelist1_bbox, Panelist1_VAD, Panelist2_bbox, Panelist2_VAD, etc.

Files with the name PanelistXX_bbox.txt (XX is from 1 to 9) include five numbers in a row, which are:
- the frame number,
- the x coordinate of left-top corner of the bounding box, 
- the y coordinate of left-top corner of the bounding box,
- the width of the bounding box and 
- the height of the bounding box.

Files with the name PanelistXX_VAD.txt (XX is from 1 to 9) include 2 numbers in a row, which are:
- the frame number,
- the speaking status; 1 for speaking, 0 for not-speaking.

The frame numbers given in PanelistXX_VAD.txt can be used to test any VAD model.
