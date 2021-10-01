#include "pch.h"
#include "torch_functions.h"

namespace ANNs {

	void ANN_regression();

	void AllCalls();
}

void plot_data(
	std::vector<double> &ax,
	std::vector<double> &ay,
	std::vector<double> &bx,
	std::vector<double> &by,
	std::string path
) {

	//for (auto a : ax) std::cout << a << std::endl;

	RGBA s1Color{ 0,0,1,1 }; // R,G,B,A
	RGBA s2Color{ 1,0,0,1 }; // R,G,B,A

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
	series1->xs = &ax;
	series1->ys = &ay;
	series1->linearInterpolation = false;
	series1->pointType = toVector(L"dots");
	series1->color = &s1Color;

	ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
	series2->xs = &bx;
	series2->ys = &by;
	series2->linearInterpolation = false;
	series2->pointType = toVector(L"dots");
	series2->color = &s2Color;

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Data");
	settings->xLabel = toVector(L"dimension 1");
	settings->yLabel = toVector(L"dimension 2");
	settings->scatterPlotSeries->push_back(series1);
	settings->scatterPlotSeries->push_back(series2);

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, path);
	DeleteImage(imageReference->image);
}

void plot_data_err(
	std::vector<double>& ax,
	std::vector<double>& ay,
	std::vector<double>& bx,
	std::vector<double>& by,
	std::vector<double>& errx,
	std::vector<double>& erry,
	std::string path
) {

	//for (auto a : ax) std::cout << a << std::endl;

	RGBA s1Color{ 0,0,1,1 }; // R,G,B,A
	RGBA s2Color{ 0,1,0,1 }; // R,G,B,A
	RGBA s3Color{ 1,0,0,1 };

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
	series1->xs = &ax;
	series1->ys = &ay;
	series1->linearInterpolation = false;
	series1->pointType = toVector(L"dots");
	series1->color = &s1Color;

	ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
	series2->xs = &bx;
	series2->ys = &by;
	series2->linearInterpolation = false;
	series2->pointType = toVector(L"dots");
	series2->color = &s2Color;

	ScatterPlotSeries* series3 = GetDefaultScatterPlotSeriesSettings();
	series3->xs = &errx;
	series3->ys = &erry;
	series3->linearInterpolation = false;
	series3->pointType = toVector(L"crosses");
	series3->color = &s3Color;

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Data");
	settings->xLabel = toVector(L"dimension 1");
	settings->yLabel = toVector(L"dimension 2");
	settings->scatterPlotSeries->push_back(series1);
	settings->scatterPlotSeries->push_back(series2);
	settings->scatterPlotSeries->push_back(series3);

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, path);
	DeleteImage(imageReference->image);
}

void plot_losses(torch::Tensor losses_t, int numepochs) {

	std::vector<double>losses (losses_t.data_ptr<float>(), losses_t.data_ptr<float>() + losses_t.numel());
	std::vector<double> epochs(numepochs);
	std::iota(std::begin(epochs), std::end(epochs), 0);

	RGBA s1Color{ 0,0,1,1 }; // R,G,B,A
	RGBA s2Color{ 1,0,0,1 }; // R,G,B,A

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSeries* series = GetDefaultScatterPlotSeriesSettings();
	series->ys = &losses;
	series->xs = &epochs;
	series->linearInterpolation = false;
	series->pointType = toVector(L"dots");
	series->color = &s1Color;

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Data");
	settings->xLabel = toVector(L"Epochs");
	settings->yLabel = toVector(L"Loss");
	settings->scatterPlotSeries->push_back(series);

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, "plots/ANN_classifier_losses.png");
	DeleteImage(imageReference->image);
}

int main(int argc, char** args) {
	
	/*
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();
	*/

	int nPerClust = 100;
	int blur = 1;

	int A[] = { 1, 1 }; // centroid category 1
	int	B[] = { 5, 1 }; // centroid category 2


	torch::Tensor ax_t = A[0] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor ay_t = A[1] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor bx_t = B[0] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor by_t = B[1] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;

	std::vector<double> ax(ax_t.data_ptr<float>(), ax_t.data_ptr<float>() + ax_t.numel());
	std::vector<double> ay(ay_t.data_ptr<float>(), ay_t.data_ptr<float>() + ay_t.numel());
	std::vector<double> bx(bx_t.data_ptr<float>(), bx_t.data_ptr<float>() + bx_t.numel());
	std::vector<double> by(by_t.data_ptr<float>(), by_t.data_ptr<float>() + by_t.numel());
	
	plot_data(ax, ay, bx, by, "plots/ANN_classifier.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });
	
	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma region creating and train the model

	torch::nn::Sequential ANNclassify(
		torch::nn::Linear(2, 1),
		torch::nn::ReLU(),
		torch::nn::Linear(1, 1),
		torch::nn::Sigmoid()
	);

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify->parameters(), learningRate);

	int numepochs = 1000;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNclassify->forward(data);

		torch::Tensor loss = torch::binary_cross_entropy(yhat, labels);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	torch::Tensor predictions = ANNclassify->forward(data);

	plot_losses(losses, numepochs);

	int errors = 0;
	bool y = false;
	bool yHat = false;

	vector<double> errx;
	vector<double> erry;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > 0.5)
			yHat = true;

		if (y != yHat) {

			errx.push_back(data[i][0].item<double>());
			erry.push_back(data[i][1].item<double>());
			errors++;
		}
	}
	std::cout << "Accuracy : "<< 100*(((nPerClust * 2)-errors)/float(nPerClust*2))<< "%" << std::endl;

	vector<double> catAx;
	vector<double> catAy;
	vector<double> catBx;
	vector<double> catBy;

	for (int i = 0; i < nPerClust * 2; i++) {
		
		if (predictions[i].item<float>() < 0.5) {
			catAx.push_back(data[i][0].item<double>());
			catAy.push_back(data[i][1].item<double>());
		}
		else {
			catBx.push_back(data[i][0].item<double>());
			catBy.push_back(data[i][1].item<double>());
		}
	}

	plot_data_err(catAx, catAy, catBx, catBy, errx, erry, "plots/ANN_classifier_pred.png");

#pragma endregion

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