# Parallel Image Quantization using K-means Clustering

## About

This is a project made for the CS 4220 GPU Computing class in Cal Poly Pomona. Note that it was made with some help from LLMs. 

**Image (color) quantization** is the process of compressing the color palette of an image to smaller set. To choose the most representative color palette, this project will use K-means clustering which is an unsupervised machine learning algorithm designed to cluster points of data. Colors most representative of the image are used by more pixels and are near each other, and are therefore a good candidate for clustering.
## File System
- `main.cu` - the main program for running image quantization.
	- Holds all of the kernels.
	- Holds the benchmarking test.
- `extract_color.py` - responsible for plotting colors for visualization.
- `examples` - has example images used by the main program.
## How to Run
- This was mainly ran on the `delta` GPUs and thus may have some issues running on other environments.

- To use `main.cu`:
	- Load the `opencv` module: `module load opencv`
	- Compile the program by running: `nvcc -o main main.cu -I $OPENCV_HOME/include/opencv4/ -L$OPENCV_HOME/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc`
	- Run the program with: `./main <image_path> <#_of_colors> [iterations]`
		- With no `[iterations]`, quantization is run without benchmarking.
		- With `[iterations]`, benchmarking is used for those many iterations.
- To use `extract_color.py`:
	- Make sure to install the necessary dependencies within `requirements.txt`.
	- Run the program with: `python extract_color.py <images>`

## Dependencies
- `main.cu` - only dependency is `opencv`.
- `extract_colors.py` - dependencies are listed in `requirements.txt`

