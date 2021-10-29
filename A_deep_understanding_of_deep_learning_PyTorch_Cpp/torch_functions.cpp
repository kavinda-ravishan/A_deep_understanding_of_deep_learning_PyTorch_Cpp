#include "torch_functions.h"

#pragma region Math_numpy_PyTorch

void Math_numpy_PyTorch::Tensor() {

	torch::Tensor s = torch::tensor(3, torch::kInt);
	torch::Tensor v = torch::tensor(
		{ 1, 2, 3 },
		torch::kInt
	);
	torch::Tensor M = torch::tensor(
		{
			{1, 2, 3},
			{4, 5, 6}
		},
		torch::kInt
	);

	for (int i = 0; i < 6; i++) {

		std::cout << *(M.data_ptr<int>() + i) << std::endl;
	}


	std::cout << s << std::endl;
	std::cout << s.item<int>() << std::endl;

	std::cout << v << std::endl;
	std::cout << v.size(0) << std::endl;

	std::cout << M << std::endl;
	std::cout << M.size(0) << std::endl;
	std::cout << M.size(1) << std::endl;

}

void Math_numpy_PyTorch::Vector_and_matrix_transpose() {

	// Vector and matrix transpose

	torch::Tensor tv = torch::randint(1, 100, { 1,3 });
	torch::Tensor tM = torch::randint(1, 100, { 2,3 });
	torch::Tensor tharray = torch::tensor({ 1, 2, 3, 4, 5 }, { torch::kFloat64 });

	std::cout << tv << std::endl;
	std::cout << tv.t() << std::endl;

	std::cout << tM << std::endl;
	std::cout << tM.t() << std::endl;

	for (int i = 0; i < 3; i++) {

		std::cout << tv[0][i].item<float>() << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 3; i++) {
			std::cout << tM[j][i].item<float>() << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for (int i = 0; i < 3; i++) {

		tv[0][i] = 1;
	}
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 3; i++) {
			tM[j][i] = 1;
		}
	}

	for (int i = 0; i < 3; i++) {

		std::cout << tv[0][i].item<float>() << " ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 3; i++) {
			std::cout << tM[j][i].item<float>() << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;


	for (int i = 0; i < 5; i++) {

		std::cout << tharray[i].item<float>() << " ";
	}
	std::cout << std::endl;
}

void Math_numpy_PyTorch::The_dot_product() {

	torch::Tensor tv1 = torch::tensor({ 1, 2, 3, 4 }, { torch::kInt64 });
	torch::Tensor tv2 = torch::tensor({ 0, 1, 0, -1 }, { torch::kInt64 });

	std::cout << torch::dot(tv1, tv2).item<int>() << std::endl;
	std::cout << torch::sum(tv1 * tv2).item<int>() << std::endl;
}

void Math_numpy_PyTorch::Matrix_multiplication() {

	torch::Tensor A = torch::randn({ 3,4 });
	torch::Tensor B = torch::randn({ 4,5 });
	torch::Tensor C = torch::randn({ 4,5 });

	std::cout << torch::matmul(A, B) << std::endl;
	std::cout << torch::matmul(B.t(), C) << std::endl;
}

void Math_numpy_PyTorch::Softmax() {

	torch::Tensor z = torch::tensor(
		{ {1}, {2}, {3} },
		{ torch::kFloat64 }
	);

	torch::Tensor sigma1 = torch::softmax(z, 0);
	std::cout << sigma1 << std::endl;

	torch::Tensor num = torch::exp(z);
	torch::Tensor denVal = torch::sum(torch::exp(z));
	torch::Tensor den = torch::full({ 3, 1 }, denVal.item<float>());
	torch::Tensor sigma2 = num / den;

	std::cout << sigma2 << std::endl;
}

void Math_numpy_PyTorch::Logarithms() {

	int size = 5;

	torch::Tensor x = torch::linspace(0.0001, 1, size);

	torch::Tensor logx = torch::log(x);
	torch::Tensor expx = torch::exp(x);

	std::cout << "Log(x) :\n" << logx << std::endl;
	std::cout << "Exp(x) :\n" << expx << std::endl;

	torch::Tensor expLogx = torch::exp(logx);
	torch::Tensor logExpx = torch::log(expx);

	std::cout << "Exp(Log(x)) - Log(Exp(x))" << std::endl;

	for (int i = 0; i < size; i++) {

		std::cout << std::setw(11) << expLogx[i].item<float>() << " - " << logExpx[i].item<float>() << std::endl;
	}
}

void Math_numpy_PyTorch::Entropyand_cross_entropy() {

	torch::Tensor p = torch::tensor({ 1, 0 }, torch::kFloat64);
	torch::Tensor q = torch::tensor({ 0.25, 0.75 }, torch::kFloat64);
	torch::Tensor H = torch::zeros({ 1 });

	for (int i = 0; i < p.sizes()[0]; i++) {

		H -= p[i] * torch::log(q[i]);
	}
	std::cout << "Cross-entropy : " << H << std::endl;


	H = -(p[0] * torch::log(q[0]) + p[1] * torch::log(q[1]));
	std::cout << "Cross-entropy : " << H << std::endl;

	H = torch::binary_cross_entropy(q, p);
	std::cout << "Cross-entropy : " << H << std::endl;

}

void Math_numpy_PyTorch::Min_maxand_argmin_argmax() {

	torch::Tensor v = torch::tensor({ 1, 20, 2, -3 }, torch::kInt);

	torch::Tensor minIndex = torch::argmin(v);
	torch::Tensor maxIndex = torch::argmax(v);

	std::cout << v << std::endl;

	std::cout << "Min index : " << minIndex.item<int>() << std::endl;
	std::cout << "Max index : " << maxIndex.item<int>() << std::endl;

	std::cout << "Min : " << v[minIndex].item<int>() << std::endl;
	std::cout << "Max : " << v[maxIndex].item<int>() << std::endl;
	std::cout << std::endl;

	torch::Tensor M = torch::tensor(
		{
			{0, 1, 10},
			{20, 8, 5}
		},
		torch::kInt
	);

	std::cout << M << std::endl;

	torch::Tensor min1 = torch::min(M);
	std::tuple<torch::Tensor, torch::Tensor> min2 = torch::min(M, 0);
	std::tuple<torch::Tensor, torch::Tensor> min3 = torch::min(M, 1);

	std::cout << min1 << std::endl;
	std::cout << std::get<0>(min2) << std::endl;
	std::cout << std::get<1>(min2) << std::endl;
	std::cout << std::get<0>(min3) << std::endl;
	std::cout << std::get<1>(min3) << std::endl;

	std::cout << torch::argmin(M) << std::endl;
	std::cout << torch::argmin(M, 0) << std::endl;
	std::cout << torch::argmin(M, 1) << std::endl;

}

void Math_numpy_PyTorch::Mean_variance() {

	torch::Tensor x = torch::tensor(
		{ 1, 2, 4, 6, 5, 4, 0 },
		torch::kFloat64
	);

	torch::Tensor mean = torch::sum(x) / x.size(0);
	torch::Tensor variance = torch::sum(torch::square(x - mean)) / (x.size(0) - torch::tensor(1, torch::kFloat64));
	torch::Tensor stdev = torch::sqrt(variance);

	std::cout << "Mean : " << torch::mean(x).item<float>() << ", " << mean.item<float>() << std::endl;
	std::cout << "Variance : " << torch::square(torch::std(x)).item<float>() << ", " << variance.item<float>() << std::endl;
	std::cout << "Standard deviation : " << torch::std(x).item<float>() << ", " << stdev.item<float>() << std::endl;
}

torch::Tensor Math_numpy_PyTorch::Random_choice(torch::Tensor a, int size) {

	int n = a.size(0);
	int c;
	torch::Tensor sample = torch::zeros(size, a.dtype());

	for (int i = 0; i < size; i++) {

		c = torch::randint(n, 1).item<int>();
		sample[i] = a[c];
	}

	return sample;
}

void Math_numpy_PyTorch::Random_samplingand_sampling_variability() {

	std::vector<int> val = { 1, 2, 3, 4, -1, -5, 0, 1, 4, 9, 6, -9, 6, -4, 2, 1 };
	int sampleSize = 5;
	int nExers = 1000;

	torch::Tensor x = torch::tensor(
		val,
		torch::kFloat64
	);

	torch::Tensor popMean = torch::mean(x);
	torch::Tensor sampleMeans = torch::zeros(nExers);
	torch::Tensor sample = torch::zeros(sampleSize);

	for (int i = 0; i < nExers; i++) {

		sample = Random_choice(x, sampleSize);
		sampleMeans[i] = torch::mean(sample);
	}

	std::cout << popMean << std::endl;
	std::cout << torch::mean(sampleMeans) << std::endl;

}

void Math_numpy_PyTorch::Reproducible_randomness_seeding() {

	std::cout << torch::randn({ 1, 5 }) << std::endl;

	torch::manual_seed(17);

	torch::Tensor a = torch::tensor({ {-1.4135, 0.2336, 0.0340, 0.3499, -0.0145} });
	torch::Tensor b = torch::tensor({ {-0.6124, -1.1835, -1.4831, 1.8004, 0.0096} });

	std::cout << a << std::endl;
	std::cout << torch::randn({ 1, 5 }) << std::endl;
	std::cout << b << std::endl;
	std::cout << torch::randn({ 1, 5 }) << std::endl;
}

void Math_numpy_PyTorch::AllCalls() {
	Tensor();
	Vector_and_matrix_transpose();
	The_dot_product();
	Matrix_multiplication();
	Softmax();
	Logarithms();
	Entropyand_cross_entropy();
	Min_maxand_argmin_argmax();
	Mean_variance();
	Random_samplingand_sampling_variability();
	Reproducible_randomness_seeding();
}

#pragma endregion

#pragma region Gradient_Descent

torch::Tensor Gradient_Descent::fx(torch::Tensor& x) {
	return (3 * x.pow(2)) - (3 * x) + 4;
}

torch::Tensor Gradient_Descent::deriv(torch::Tensor& x) {
	return (6 * x) - 3;
}

void Gradient_Descent::Gradient_descent_in_1D(torch::Tensor(*deriv)(torch::Tensor& x)) {

	int x_len = 2001;
	torch::Tensor x = torch::linspace(-2, 2, x_len);
	torch::Tensor localMin = x[torch::randint(0, x_len, 1).item<int>()];

	std::cout << localMin << std::endl;

	int training_epochs = 100;
	torch::Tensor learning_rate = torch::tensor(0.01, torch::kFloat64);
	torch::Tensor grad = torch::tensor(0, torch::kFloat64);

	for (int i = 0; i < training_epochs; i++) {

		grad = deriv(localMin);
		localMin = localMin - (learning_rate * grad);
	}

	std::cout << localMin << std::endl;
}

void Gradient_Descent::AllCalls() {

	Gradient_descent_in_1D(deriv);
}

#pragma endregion

#pragma region ANNs

void ANNs::plot_data(
	std::vector<double>& ax,
	std::vector<double>& ay,
	std::vector<double>& bx,
	std::vector<double>& by,
	std::string path
) {

	//for (auto a : ax) std::cout << a << std::endl;

	RGBA s1Color{ 0,0,1,1 }; // R,G,B,A
	RGBA s2Color{ 0,1,0,1 }; // R,G,B,A

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 800;
	settings->height = 600;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Data");
	settings->xLabel = toVector(L"dimension 1");
	settings->yLabel = toVector(L"dimension 2");

	if (ax.size() != 0 && ay.size() != 0) {
		ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
		settings->scatterPlotSeries->push_back(series1);
		series1->xs = &ax;
		series1->ys = &ay;
		series1->linearInterpolation = false;
		series1->pointType = toVector(L"dots");
		series1->color = &s1Color;
	}

	if (bx.size() != 0 && by.size() != 0) {
		ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
		settings->scatterPlotSeries->push_back(series2);
		series2->xs = &bx;
		series2->ys = &by;
		series2->linearInterpolation = false;
		series2->pointType = toVector(L"dots");
		series2->color = &s2Color;
	}

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, path);
	DeleteImage(imageReference->image);
}

void ANNs::plot_data_err(
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


	ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
	settings->width = 800;
	settings->height = 600;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = toVector(L"Data");
	settings->xLabel = toVector(L"dimension 1");
	settings->yLabel = toVector(L"dimension 2");

	if (ax.size() != 0 && ay.size() != 0) {
		ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
		settings->scatterPlotSeries->push_back(series1);
		series1->xs = &ax;
		series1->ys = &ay;
		series1->linearInterpolation = false;
		series1->pointType = toVector(L"dots");
		series1->color = &s1Color;
	}

	if (bx.size() != 0 && by.size() != 0) {
		ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
		settings->scatterPlotSeries->push_back(series2);
		series2->xs = &bx;
		series2->ys = &by;
		series2->linearInterpolation = false;
		series2->pointType = toVector(L"dots");
		series2->color = &s2Color;
	}

	if (errx.size() != 0 && erry.size() != 0) {
		ScatterPlotSeries* series3 = GetDefaultScatterPlotSeriesSettings();
		settings->scatterPlotSeries->push_back(series3);
		series3->xs = &errx;
		series3->ys = &erry;
		series3->linearInterpolation = false;
		series3->pointType = toVector(L"crosses");
		series3->color = &s3Color;
	}

	DrawScatterPlotFromSettings(imageReference, settings);

	vector<double>* pngdata = ConvertToPNG(imageReference->image);
	WriteToFile(pngdata, path);
	DeleteImage(imageReference->image);
}

void ANNs::plot_losses(torch::Tensor losses_t, int numepochs, std::string path) {

	std::vector<double>losses(losses_t.data_ptr<float>(), losses_t.data_ptr<float>() + losses_t.numel());
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
	WriteToFile(pngdata, path);
	DeleteImage(imageReference->image);
}

std::vector<std::string> ANNs::split(const std::string& s, char delimiter) {

	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

void ANNs::read_csv(
	std::string path,
	std::vector<std::vector<std::string>>& data,
	std::vector<std::string>& colNames
)
{

	std::string text;
	std::string headstr;
	std::vector<std::string> filestr;
	std::ifstream file;
	try {

		file.open(path);
		if (!file.is_open()) throw - 1;

		getline(file, headstr);
		while (getline(file, text)) {

			filestr.push_back(text);
		}
		file.close();
	}
	catch (...) {

		std::cout << "Can not open the file." << std::endl;
		file.close();
	}

	colNames = split(headstr, ',');

	int numCols = colNames.size();

	for (int i = 0; i < numCols; i++) {

		data.push_back(std::vector<std::string>());
	}

	for (std::string line : filestr) {

		std::vector<std::string> row = split(line, ',');

		for (int i = 0; i < numCols; i++) {

			data[i].push_back(row[i]);
		}
	}
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

void ANNs::ANN_classification() {

#pragma region Create data

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

	plot_data(ax, ay, bx, by, "plots/ANN_classifier_1.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma endregion

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

	plot_losses(losses, numepochs, "plots/ANN_classifier_3_losses.png");

	int errors = 0;
	bool y = false;
	bool yHat = false;

	vector<double> errx;
	vector<double> erry;

	float threshold = 0.5;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > threshold)
			yHat = true;

		if (y != yHat) {

			errx.push_back(data[i][0].item<double>());
			erry.push_back(data[i][1].item<double>());
			errors++;
		}
	}
	std::cout << "Accuracy : " << 100 * (((nPerClust * 2) - errors) / float(nPerClust * 2)) << "%" << std::endl;

	vector<double> catAx;
	vector<double> catAy;
	vector<double> catBx;
	vector<double> catBy;

	for (int i = 0; i < nPerClust * 2; i++) {

		if (predictions[i].item<float>() <= threshold) {
			catAx.push_back(data[i][0].item<double>());
			catAy.push_back(data[i][1].item<double>());
		}
		else {
			catBx.push_back(data[i][0].item<double>());
			catBy.push_back(data[i][1].item<double>());
		}
	}

	plot_data_err(catAx, catAy, catBx, catBy, errx, erry, "plots/ANN_classifier_2_pred.png");

#pragma endregion
}

void ANNs::Multilayer_ANN_classification() {

#pragma region Create data

	int nPerClust = 100;
	int blur = 1;

	int A[] = { 1, 3 }; // centroid category 1
	int	B[] = { 1, -2 }; // centroid category 2


	torch::Tensor ax_t = A[0] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor ay_t = A[1] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor bx_t = B[0] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;
	torch::Tensor by_t = B[1] + torch::randn({ nPerClust }, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) * blur;

	std::vector<double> ax(ax_t.data_ptr<float>(), ax_t.data_ptr<float>() + ax_t.numel());
	std::vector<double> ay(ay_t.data_ptr<float>(), ay_t.data_ptr<float>() + ay_t.numel());
	std::vector<double> bx(bx_t.data_ptr<float>(), bx_t.data_ptr<float>() + bx_t.numel());
	std::vector<double> by(by_t.data_ptr<float>(), by_t.data_ptr<float>() + by_t.numel());

	plot_data(ax, ay, bx, by, "plots/Multilayer_ANN_classifier_1.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma endregion

#pragma region creating and train the model

	torch::nn::Sequential ANNclassify(
		torch::nn::Linear(2, 16),
		torch::nn::ReLU(),
		torch::nn::Linear(16, 1),
		torch::nn::ReLU(),
		torch::nn::Linear(1, 1)
	);

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify->parameters(), learningRate);

	int numepochs = 1000;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNclassify->forward(data);

		torch::Tensor loss = torch::binary_cross_entropy_with_logits(yhat, labels);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	torch::Tensor predictions = ANNclassify->forward(data);

	plot_losses(losses, numepochs, "plots/Multilayer_ANN_classifier_3_losses.png");

	int errors = 0;
	bool y = false;
	bool yHat = false;

	vector<double> errx;
	vector<double> erry;

	float threshold = 0;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > threshold)
			yHat = true;

		if (y != yHat) {

			errx.push_back(data[i][0].item<double>());
			erry.push_back(data[i][1].item<double>());
			errors++;
		}
	}

	std::cout << "Accuracy : " << 100 * (((nPerClust * 2) - errors) / float(nPerClust * 2)) << "%" << std::endl;

	vector<double> catAx;
	vector<double> catAy;
	vector<double> catBx;
	vector<double> catBy;

	for (int i = 0; i < nPerClust * 2; i++) {

		if (predictions[i].item<float>() <= threshold) {
			catAx.push_back(data[i][0].item<double>());
			catAy.push_back(data[i][1].item<double>());
		}
		else {
			catBx.push_back(data[i][0].item<double>());
			catBy.push_back(data[i][1].item<double>());
		}
	}


	plot_data_err(catAx, catAy, catBx, catBy, errx, erry, "plots/Multilayer_ANN_classifier_2_pred.png");

#pragma endregion
}

void ANNs::ANN_iris_dataset() {

#pragma region read_and_prepare_data

	std::string path = "datasets/iris.csv";
	std::vector<std::vector<std::string>> data;
	std::vector<std::string> colNames;

	read_csv(path, data, colNames);

	int numCols = data.size();

	std::vector<std::vector<float>> vX;
	for (int i = 0; i < numCols - 1; i++) {

		vX.push_back(std::vector<float>());
	}

	for (int i = 0; i < numCols - 1; i++) {

		for (std::string value : data[i]) {

			vX[i].push_back(std::stod(value));
		}
	}

	int column = numCols - 1;
	std::vector<int> vy;
	std::vector<std::string> categories;
	bool is_category_exists = false;

	for (std::string label : data[column]) {

		is_category_exists = false;
		int i = 0;
		for (std::string category : categories) {

			if (category == label) {

				is_category_exists = true;
				break;
			}
			i++;
		}

		if (!is_category_exists) categories.push_back(label);

		vy.push_back(i);
	}

	torch::Tensor X0 = torch::tensor(vX[0], torch::kFloat);
	torch::Tensor X1 = torch::tensor(vX[1], torch::kFloat);
	torch::Tensor X2 = torch::tensor(vX[2], torch::kFloat);
	torch::Tensor X3 = torch::tensor(vX[3], torch::kFloat);

	torch::Tensor X = torch::stack({ X0, X1, X2, X3 }, 1);

	torch::Tensor y = torch::tensor(vy, torch::kInt64);
#pragma endregion

#pragma region create_model

	torch::nn::Sequential ANNclassify(
		torch::nn::Linear(4, 64),
		torch::nn::ReLU(),
		torch::nn::Linear(64, 64),
		torch::nn::ReLU(),
		torch::nn::Linear(64, 3)
	);

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify->parameters(), learningRate);

	int numepochs = 1000;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);
	torch::Tensor ongoingAcc = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {

		torch::Tensor yhat = ANNclassify->forward(X);

		torch::Tensor loss = torch::nn::functional::cross_entropy(yhat, y);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		torch::Tensor pred = torch::argmax(yhat, 1);
		int numPreds = pred.size(0);
		int num_correct_preds = 0;
		for (int i = 0; i < numPreds; i++) {

			if (pred[i].item<int>() == vy[i]) num_correct_preds++;
		}

		ongoingAcc[i] = (num_correct_preds * 100) / float(numPreds);
	}


#pragma endregion

#pragma region predictions

	torch::Tensor pred = torch::argmax(ANNclassify->forward(X), 1);

	int numPreds = pred.size(0);
	int num_correct_preds = 0;
	for (int i = 0; i < numPreds; i++) {

		if (pred[i].item<int>() == vy[i]) num_correct_preds++;
	}

	std::cout << "Accuracy : " << (num_correct_preds * 100) / float(numPreds) << "%" << std::endl;

	plot_losses(losses, numepochs, "plots/Iris_dataset_losses.png");
	plot_losses(ongoingAcc, numepochs, "plots/Iris_dataset_accuracy.png");
#pragma endregion

}

ANNs::ANNclassifyClass::ANNclassifyClass() {

	input = register_module("input", torch::nn::Linear(2, 1));
	output = register_module("output", torch::nn::Linear(1, 1));
}

torch::Tensor ANNs::ANNclassifyClass::forward(torch::Tensor x) {

	x = input->forward(x);
	x = torch::relu(x);
	x = output->forward(x);
	x = torch::sigmoid(x);

	return x;
}

void ANNs::ANN_class_classification() {

#pragma region Create data

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

	plot_data(ax, ay, bx, by, "plots/ANN_classifier_1.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma endregion

#pragma region creating and train the model

	/*torch::nn::Sequential ANNclassify(
		torch::nn::Linear(2, 1),
		torch::nn::ReLU(),
		torch::nn::Linear(1, 1),
		torch::nn::Sigmoid()
	);*/

	ANNs::ANNclassifyClass ANNclassify;

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify.parameters(), learningRate);

	int numepochs = 1000;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNclassify.forward(data);

		torch::Tensor loss = torch::binary_cross_entropy(yhat, labels);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	torch::Tensor predictions = ANNclassify.forward(data);

	plot_losses(losses, numepochs, "plots/ANN_classifier_3_losses.png");

	int errors = 0;
	bool y = false;
	bool yHat = false;

	vector<double> errx;
	vector<double> erry;

	float threshold = 0.5;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > threshold)
			yHat = true;

		if (y != yHat) {

			errx.push_back(data[i][0].item<double>());
			erry.push_back(data[i][1].item<double>());
			errors++;
		}
	}
	std::cout << "Accuracy : " << 100 * (((nPerClust * 2) - errors) / float(nPerClust * 2)) << "%" << std::endl;

	vector<double> catAx;
	vector<double> catAy;
	vector<double> catBx;
	vector<double> catBy;

	for (int i = 0; i < nPerClust * 2; i++) {

		if (predictions[i].item<float>() <= threshold) {
			catAx.push_back(data[i][0].item<double>());
			catAy.push_back(data[i][1].item<double>());
		}
		else {
			catBx.push_back(data[i][0].item<double>());
			catBy.push_back(data[i][1].item<double>());
		}
	}

	plot_data_err(catAx, catAy, catBx, catBy, errx, erry, "plots/ANN_classifier_2_pred.png");

#pragma endregion
}

ANNs::ANN_ModuleDict::ANN_ModuleDict(int nUnits, int nLayers) {

	for (int i = 1; i <= nLayers; i++) {

		std::string layerName = "hidden";
		layerName.append(std::to_string(i));
		layersNames.push_back(layerName);
	}

	torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict;

	ordereddict.insert("input", torch::nn::Linear(2, nUnits).ptr());

	for (std::string lstr : layersNames) {

		ordereddict.insert(lstr, torch::nn::Linear(nUnits, nUnits).ptr());
	}

	ordereddict.insert("output", torch::nn::Linear(nUnits, 1).ptr());

	torch::nn::ModuleDict dict(ordereddict);
	ann_dict = register_module("ann_dict", dict);
}

torch::Tensor ANNs::ANN_ModuleDict::forward(torch::Tensor x) {

	x = ann_dict["input"]->as<torch::nn::Linear>()->forward(x);
	x = torch::relu(x);

	for (std::string lstr : layersNames) {

		x = torch::relu(ann_dict[lstr]->as<torch::nn::Linear>()->forward(x));
	}

	x = ann_dict["output"]->as<torch::nn::Linear>()->forward(x);
	x = torch::sigmoid(x);

	return x;
}

void ANNs::ANN_class_ModuleDict_classification() {

#pragma region Create data

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

	plot_data(ax, ay, bx, by, "plots/ANN_classifier_ModuleDict_1.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma endregion

#pragma region creating and train the model

	ANNs::ANN_ModuleDict ANNclassify(10, 5);

	std::cout << ANNclassify << std::endl;

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify.parameters(), learningRate);

	int numepochs = 1000;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNclassify.forward(data);

		torch::Tensor loss = torch::binary_cross_entropy(yhat, labels);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	torch::Tensor predictions = ANNclassify.forward(data);

	plot_losses(losses, numepochs, "plots/ANN_classifier_ModuleDict_3_losses.png");

	int errors = 0;
	bool y = false;
	bool yHat = false;

	vector<double> errx;
	vector<double> erry;

	float threshold = 0.5;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > threshold)
			yHat = true;

		if (y != yHat) {

			errx.push_back(data[i][0].item<double>());
			erry.push_back(data[i][1].item<double>());
			errors++;
		}
	}
	std::cout << "Accuracy : " << 100 * (((nPerClust * 2) - errors) / float(nPerClust * 2)) << "%" << std::endl;

	vector<double> catAx;
	vector<double> catAy;
	vector<double> catBx;
	vector<double> catBy;

	for (int i = 0; i < nPerClust * 2; i++) {

		if (predictions[i].item<float>() <= threshold) {
			catAx.push_back(data[i][0].item<double>());
			catAy.push_back(data[i][1].item<double>());
		}
		else {
			catBx.push_back(data[i][0].item<double>());
			catBy.push_back(data[i][1].item<double>());
		}
	}

	plot_data_err(catAx, catAy, catBx, catBy, errx, erry, "plots/ANN_classifier_ModuleDict_2_pred.png");

#pragma endregion
}

void ANNs::AllCalls() {

	ANN_regression();
	ANN_classification();
	Multilayer_ANN_classification();
	ANN_iris_dataset();
	ANN_class_classification();
	ANNs::ANN_class_ModuleDict_classification();
}

#pragma endregion