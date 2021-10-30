#include "pch.h"

int main(int argc, char** args) {

	torch::Tensor t = torch::randn({5, 5});
	cv::Mat i(cv::Size(1, 49), CV_64FC1);
	
	std::cout << t << std::endl;
	std::cout << i << std::endl;
	
	return 0;
}

