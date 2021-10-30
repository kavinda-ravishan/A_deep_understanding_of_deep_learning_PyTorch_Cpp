#include "pch.h"

#define DEFAULT_WIDTH 720
#define DEFAULT_HEIGHT 1280
#define IMG_SIZE 512

cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model)
{
	double alpha = 0.4;
	double beta = 1 - alpha;
	cv::Mat frame_copy, dst;
	std::vector<torch::jit::IValue> input;
	std::vector<double> mean = { 0.406, 0.456, 0.485 };
	std::vector<double> std = { 0.225, 0.224, 0.229 };

	cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
	frame_copy = frame;
	frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);

	//cv to tensor
	torch::Tensor frame_tensor = torch::from_blob(frame.data, { 1, IMG_SIZE, IMG_SIZE, 3 });
	frame_tensor = frame_tensor.permute({ 0, 3, 1, 2 }); // { 1, 3, IMG_SIZE, IMG_SIZE}

	frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
	frame_tensor = frame_tensor.to(torch::kCPU);

	input.push_back(frame_tensor);

	auto pred = model.forward(input).toTensor().detach().to(torch::kCPU);
	pred = pred.mul(100).clamp(0, 255).to(torch::kU8);

	cv::Mat output_mat(cv::Size(IMG_SIZE, IMG_SIZE), CV_8UC1, pred.data_ptr());
	cv::cvtColor(output_mat, output_mat, cv::COLOR_GRAY2RGB);
	cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_TWILIGHT_SHIFTED);

	cv::addWeighted(frame_copy, alpha, output_mat, beta, 0.0, dst);
	cv::resize(dst, dst, cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT));

	return dst;
}

torch::jit::Module load_model(std::string model_name)
{
	std::string dir = "./models/" + model_name;
	torch::jit::Module module = torch::jit::load(dir);
	module.to(torch::kCPU);
	module.eval();
	std::cout << "MODEL LOADED.." << std::endl;

	return module;
}


int main(int argc, char** args) {

	torch::jit::script::Module module;

	cv::VideoCapture cap;
	cv::Mat frame;
	cap.open("./videos/driving_1.mp4");

	if (!cap.isOpened()) {
		std::cout << "Can not open the video file!!" << std::endl;
	}
	std::cout << "Video file opened.." << std::endl;


	try {
		module = load_model("quantized_lanesNet.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "ERROR: MODEL DID NOT LOAD!!\n";
	}

	std::cout << "Press Esc to terminate" << std::endl;

	while (true)
	{
		cap.read(frame);
		if (frame.empty()) {
			std::cerr << "Error: Blank frame!!\n";
			break;
		}

		frame = frame_prediction(frame, module);
		cv::imshow("video", frame);

		if (cv::waitKey(10) == 27) break;
	}

	return 0;
}

