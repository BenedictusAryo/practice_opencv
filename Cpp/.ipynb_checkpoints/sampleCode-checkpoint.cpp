#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{

	// Read image in GrayScale mode
	Mat image = imread("C:\\Users\\Benedict\\OneDrive\\Github\\practice_opencv\\Cpp\\boy.jpg");

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);				// Show our image inside it.
	waitKey(0);										// Wait for a keystroke in the window

	return 0;
}
