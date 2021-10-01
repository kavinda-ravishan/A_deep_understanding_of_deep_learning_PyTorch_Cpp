#include "pch.h"
#include "torch_functions.h"

namespace ANNs {

	void ANN_regression();

	void AllCalls();
}


int main(int argc, char** args) {
	
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();

	return 0;
}

void ANNs::ANN_regression() {

#pragma region Create data

	int N = 40;

	torch::Tensor x = torch::randn({ N,1 }, torch::kFloat32);
	torch::Tensor y = x + torch::randn({ N,1 }, torch::kFloat32) / 2;

#pragma endregion

#pragma region creating and train the model

	torch::nn::Sequential ANNReg(
		torch::nn::Linear(1, 1),
		torch::nn::ReLU(),
		torch::nn::Linear(1, 1)
	);

	float learningRate = 0.05;
	torch::optim::SGD optimizer(ANNReg->parameters(), learningRate);

	int numepochs = 500;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNReg->forward(x);

		torch::Tensor loss = torch::mse_loss(yhat, y);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	torch::Tensor predictions = ANNReg->forward(x);

	vector<double>  xVec(N);
	vector<double>  yVec(N);
	vector<double>  predVec(N);

	for (int i = 0; i < N; i++) {

		xVec[i] = x[i].item<double>();
		yVec[i] = y[i].item<double>();
		predVec[i] = predictions[i].item<double>();
	}

	for (int i = 0; i < N; i++) {

		std::cout << xVec[i] << ", " << yVec[i] << ", " << predVec[i] << std::endl;
	}
#pragma endregion

#pragma region plot data

	RGBA s1Color{ 0,0,1,1 }; // R,G,B,A
	RGBA s2Color{ 1,0,0,1 }; // R,G,B,A

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
	series1->xs = &xVec;
	series1->ys = &yVec;
	series1->linearInterpolation = false;
	series1->pointType = toVector(L"dots");
	series1->color = &s1Color;

	ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
	series2->xs = &xVec;
	series2->ys = &predVec;
	series2->linearInterpolation = false;
	series2->pointType = toVector(L"dots");
	series2->color = &s2Color;

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Linear regression");
	settings->xLabel = toVector(L"X axis");
	settings->yLabel = toVector(L"Y axis");
	settings->scatterPlotSeries->push_back(series1);
	settings->scatterPlotSeries->push_back(series2);

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, "plots/ANN_Linear_regression.png");
	DeleteImage(imageReference->image);
#pragma endregion
}

void ANNs::AllCalls() {

	ANN_regression();
}