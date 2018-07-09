# MSL1CFT
This MATLAB code implements the proposed method (MSL1CFT)in the paper: Correlation Filter Tracker based on Sparse Regularization, which has been published by JVCIR recently. 


Installation:
To be able to run the "mexResize" function, try to use either one of the included mex-files or compile one of your own. OpenCV is needed for this. The existing compile scripts "compilemex.m" or "compilemex_win.m" can be modified for this purpose.

Instructions:
1) Run the "run_tracker.m" script in MATLAB.
or you can download the OTB50 datasets and OTB2013 toolkit. In our codes, we provide the interface function: run_MSL1CFTplus.m 

Contact:
Zhangjian ji

jizhangjian@sxu.edu.cn


Citation

If you find these code useful, please consider to cite our paper：

@article{ji2018,
  title={Correlation Filter Tracker based on Sparse Regularization},
  author={Zhangjian Ji and Weiqiang Wang},
  journal={Journal of Visual Communication and Image Representation},
  year={2018},
}
