#include "pch.h"
#include "torch_functions.h"

void ALL() {
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();
	Overfittingand_cross_validation::ALLCalls();
}

namespace Regularization {
	struct IrisDataSet : torch::data::datasets::Dataset<IrisDataSet>
	{
	public:
		explicit IrisDataSet(const torch::Tensor& X, const torch::Tensor& y) :X_(X), y_(y) {}
		torch::data::Example<> get(size_t index) override
		{
			return { X_[index], y_[index] };
		}
		torch::optional<size_t> size() const override
		{
			return X_.size(0);
		}
		const torch::Tensor& X() const
		{
			return X_;
		}
		const torch::Tensor& y() const
		{
			return y_;
		}
	private:
		torch::Tensor X_;
		torch::Tensor y_;
	};

	std::pair<int, int> train_test_split(
		const std::vector<std::vector<float>>& X,
		const std::vector<int>& y,
		std::vector<std::vector<float>>& X_train,
		std::vector<std::vector<float>>& X_test,
		std::vector<int>& y_train,
		std::vector<int>& y_test,
		float train_percentage,
		int num_data_points
	);

	std::vector<std::string> split(const std::string& s, char delimiter);

	int read_csv(
		const std::string& path,
		std::vector<std::vector<std::string>>& data,
		std::vector<std::string>& colNames
	);

	int read_csv(
		const std::string& root,
		std::vector<std::vector<float>>& X,
		std::vector<int>& y,
		std::vector<std::string>& colNames
	);

	float eval_model(torch::nn::Sequential& model, const torch::Tensor& X, const torch::Tensor& y);

	void train_and_eval_modes();

	void Dropout_regularization();

	void plot_data(
		std::vector<double>& ax,
		std::vector<double>& ay,
		std::vector<double>& bx,
		std::vector<double>& by,
		const std::string& path
	);

	std::pair<int, int> train_test_split(
		const float train_percentage,
		const torch::Tensor& X,
		const torch::Tensor& y,
		torch::Tensor& X_train,
		torch::Tensor& y_train,
		torch::Tensor& X_test,
		torch::Tensor& y_test
	);

	struct DataLoader : torch::data::datasets::Dataset<DataLoader>
	{
	public:
		explicit DataLoader(const torch::Tensor& X, const torch::Tensor& y) :X_(X), y_(y) {}
		torch::data::Example<> get(size_t index) override
		{
			return { X_[index], y_[index] };
		}
		torch::optional<size_t> size() const override
		{
			return X_.size(0);
		}
		const torch::Tensor& X() const
		{
			return X_;
		}
		const torch::Tensor& y() const
		{
			return y_;
		}
	private:
		torch::Tensor X_;
		torch::Tensor y_;
	};
	
	class Net : public torch::nn::Module {
	private:
		torch::nn::Linear input{ nullptr }, hidden{ nullptr }, output{ nullptr };
		double dr;
	public:
		Net(float dropoutRate):dr(dropoutRate)
		{
			input = register_module("input", torch::nn::Linear(2, 128));
			hidden = register_module("hidden", torch::nn::Linear(128, 128));
			output = register_module("output", torch::nn::Linear(128, 1));
		}
		torch::Tensor forward(torch::Tensor x)
		{
			x = torch::relu(input->forward(x));
			x = torch::dropout(x, dr, this->is_training());

			x = torch::relu(hidden->forward(x));
			x = torch::dropout(x, dr, this->is_training());

			x = output->forward(x);

			return x;
		}
	};

	void Regular_dropout();

	void L2_regularization();

	void AllCalls();
}

int main(int argc, char** args) {

	Regularization::L2_regularization();

	return 0;
}

#pragma region Regularization

std::pair<int, int> Regularization::train_test_split(
	const std::vector<std::vector<float>>& X,
	const std::vector<int>& y,
	std::vector<std::vector<float>>& X_train,
	std::vector<std::vector<float>>& X_test,
	std::vector<int>& y_train,
	std::vector<int>& y_test,
	float train_percentage,
	int num_data_points
)
{
	auto randIndex = torch::randperm(num_data_points);

	int train_size = int(num_data_points * train_percentage);
	int test_size = num_data_points - train_size;

	for (int i = 0; i < X.size(); i++)
	{
		X_train.push_back(std::vector<float>(train_size));
	}
	for (int i = 0; i < X.size(); i++)
	{
		X_test.push_back(std::vector<float>(test_size));
	}

	y_train.resize(train_size);
	y_test.resize(test_size);

	int randRow = 0;

	// train set
	for (int row = 0; row < train_size; row++)
	{
		randRow = randIndex[row].item<int>();

		for (int col = 0; col < X.size(); col++)
		{
			X_train[col][row] = X[col][randRow];
		}
		y_train[row] = y[randRow];
	}

	//test set
	for (int row = train_size; row < num_data_points; row++)
	{
		randRow = randIndex[row].item<int>();

		for (int col = 0; col < X.size(); col++)
		{
			X_test[col][row - train_size] = X[col][randRow];
		}
		y_test[row - train_size] = y[randRow];
	}

	return { train_size, test_size };
}

std::vector<std::string> Regularization::split(const std::string& s, char delimiter) {

	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

int Regularization::read_csv(
	const std::string& path,
	std::vector<std::vector<std::string>>& data,
	std::vector<std::string>& colNames
)
{
	std::string text;
	std::string headstr;
	std::vector<std::string> filestr;
	std::ifstream file;
	int size = 0;
	try {

		file.open(path);
		if (!file.is_open()) throw - 1;

		getline(file, headstr);
		while (getline(file, text)) {

			filestr.push_back(text);
			size++;
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

		data.push_back(std::vector<std::string>(size));
	}

	int row_i = 0;
	for (std::string line : filestr) {

		std::vector<std::string> row = split(line, ',');

		for (int i = 0; i < numCols; i++) {

			data[i][row_i] = row[i];
		}
		row_i++;
	}

	return size;
}

int Regularization::read_csv(
	const std::string& root,
	std::vector<std::vector<float>>& X,
	std::vector<int>& y,
	std::vector<std::string>& colNames
)
{
	std::vector<std::vector<std::string>> data;

	int size = read_csv(root, data, colNames);

	int numCols = colNames.size();

	for (int i = 0; i < numCols - 1; i++) {

		X.push_back(std::vector<float>(size));
	}

	int row_i;
	for (int i = 0; i < numCols - 1; i++) {
		row_i = 0;
		for (std::string value : data[i]) {

			X[i][row_i] = std::stod(value);
			row_i++;
		}
	}

	int column = numCols - 1;
	std::vector<std::string> categories;
	bool is_category_exists = false;
	y.resize(size);

	row_i = 0;
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

		y[row_i] = i;
		row_i++;
	}

	return size;
}

float Regularization::eval_model(torch::nn::Sequential& model, const torch::Tensor& X, const torch::Tensor& y)
{
	torch::Tensor y_pred;

	model->eval();
	{
		torch::NoGradGuard no_grad;
		y_pred = torch::argmax(model->forward(X), 1);
	}

	int numPreds = y_pred.size(0);
	int num_correct_preds = 0;
	for (int i = 0; i < numPreds; i++)
	{
		if (y_pred[i].item<int>() == y[i].item<int>()) num_correct_preds++;
	}

	return (num_correct_preds * 100) / float(numPreds);
}

void Regularization::train_and_eval_modes()
{

#pragma region Data preprocessing

	std::vector<std::vector<float>> X;
	std::vector<int> y;
	std::vector<std::string> colNames;

	std::string root = "./datasets/iris.csv";

	int num_data_points = read_csv(root, X, y, colNames);

	std::vector<std::vector<float>> X_train;
	std::vector<std::vector<float>> X_test;
	std::vector<int> y_train;
	std::vector<int> y_test;

	float train_percentage = 0.8;

	auto train_test_sizes = train_test_split(X, y, X_train, X_test, y_train, y_test, train_percentage, num_data_points);

	torch::Tensor X_train_t = torch::stack(
		{
			torch::tensor(X_train[0], torch::kFloat),
			torch::tensor(X_train[1], torch::kFloat),
			torch::tensor(X_train[2], torch::kFloat),
			torch::tensor(X_train[3], torch::kFloat)
		}, 1);

	torch::Tensor X_test_t = torch::stack(
		{
			torch::tensor(X_test[0], torch::kFloat),
			torch::tensor(X_test[1], torch::kFloat),
			torch::tensor(X_test[2], torch::kFloat),
			torch::tensor(X_test[3], torch::kFloat)
		}, 1);

	torch::Tensor y_train_t = torch::tensor(y_train, torch::kInt64);
	torch::Tensor y_test_t = torch::tensor(y_test, torch::kInt64);
#pragma endregion

	auto train_dataset = IrisDataSet(X_train_t, y_train_t).map(torch::data::transforms::Stack<>());

	int batchSize = 4;
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batchSize);


	torch::nn::Sequential ANNclassify(
		torch::nn::Linear(4, 64),
		torch::nn::ReLU(),
		torch::nn::Linear(64, 64),
		torch::nn::ReLU(),
		torch::nn::Linear(64, 3)
	);

	float learningRate = 0.01;
	torch::optim::SGD optimizer(ANNclassify->parameters(), learningRate);

	int numepochs = 20;

	for (int i = 0; i < numepochs; i++) {

		ANNclassify->train(true);
		for (auto& batch : *train_loader)
		{
			torch::Tensor yhat = ANNclassify->forward(batch.data);

			torch::Tensor loss = torch::nn::functional::cross_entropy(yhat, batch.target);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		std::cout << "Epoch : " << setw(3) << std::setprecision(4) << i << " :: ";
		std::cout << "Training Accuracy : " << setw(6) << eval_model(ANNclassify, X_train_t, y_train_t) << "%" << ", ";
		std::cout << "Testing Accuracy  : " << setw(6) << eval_model(ANNclassify, X_test_t, y_test_t) << "%" << std::endl;
	}

	std::cout << std::setprecision(10);
	std::cout << "Training Accuracy : " << eval_model(ANNclassify, X_train_t, y_train_t) << "%" << ", ";
	std::cout << "Testing Accuracy  : " << eval_model(ANNclassify, X_test_t, y_test_t) << "%" << std::endl;

}

void Regularization::Dropout_regularization()
{
	float prob = 0.5;
	torch::nn::Dropout dropout(torch::nn::DropoutOptions().p(prob));

	torch::Tensor x = torch::ones({ 1,10 }, { torch::kFloat });
	torch::Tensor y = dropout(x);

	std::cout << x << std::endl;
	std::cout << y << std::endl;

	// -- //
	dropout->eval();
	y = dropout(x);
	std::cout << y << std::endl;

	dropout->train();
	y = dropout(x);
	std::cout << y << std::endl;

	// -- //
	y = torch::nn::functional::dropout(x);
	std::cout << y << std::endl;

	y = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().training(false));
	std::cout << y << std::endl;
}

void Regularization::plot_data(
	std::vector<double>& ax,
	std::vector<double>& ay,
	std::vector<double>& bx,
	std::vector<double>& by,
	const std::string& path
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

std::pair<int, int> Regularization::train_test_split(
	const float train_percentage,
	const torch::Tensor& X,
	const torch::Tensor& y,
	torch::Tensor& X_train,
	torch::Tensor& y_train,
	torch::Tensor& X_test,
	torch::Tensor& y_test
)
{
	const int num_data_points = X.size(0);
	const int train_size = int(num_data_points * train_percentage);
	const int test_size = num_data_points - train_size;

	torch::Tensor randIndex = torch::randperm(num_data_points);

	X_train = torch::zeros({ train_size, X.size(1) }, torch::TensorOptions().dtype(torch::kFloat));
	X_test = torch::zeros({ test_size, X.size(1) }, torch::TensorOptions().dtype(torch::kFloat));
	y_train = torch::zeros({ train_size, y.size(1) }, torch::TensorOptions().dtype(torch::kFloat));
	y_test = torch::zeros({ test_size, y.size(1) }, torch::TensorOptions().dtype(torch::kFloat));

	for (int i = 0; i < train_size; i++)
	{
		X_train[i] = X[randIndex[i].item<int>()];
		y_train[i] = y[randIndex[i].item<int>()];
	}

	for (int i = train_size; i < num_data_points; i++)
	{
		X_test[i - train_size] = X[randIndex[i].item<int>()];
		y_test[i - train_size] = y[randIndex[i].item<int>()];
	}

	return { train_size, test_size };
}

void Regularization::Regular_dropout()
{
#pragma region Create data
	const int nPerClust = 200;
	const int num_data_points = 2 * nPerClust;

	const int r1 = 10;
	const int r2 = 15;

	torch::Tensor th = torch::linspace(0, 4 * M_PI, nPerClust);

	torch::Tensor ax_t = r1 * torch::cos(th) + torch::randn({ nPerClust }) * 3;
	torch::Tensor ay_t = r1 * torch::sin(th) + torch::randn({ nPerClust });
	torch::Tensor bx_t = r2 * torch::cos(th) + torch::randn({ nPerClust }) * 3;
	torch::Tensor by_t = r2 * torch::sin(th) + torch::randn({ nPerClust });

	std::vector<double> ax(ax_t.data_ptr<float>(), ax_t.data_ptr<float>() + ax_t.numel());
	std::vector<double> ay(ay_t.data_ptr<float>(), ay_t.data_ptr<float>() + ay_t.numel());
	std::vector<double> bx(bx_t.data_ptr<float>(), bx_t.data_ptr<float>() + bx_t.numel());
	std::vector<double> by(by_t.data_ptr<float>(), by_t.data_ptr<float>() + by_t.numel());

	plot_data(ax, ay, bx, by, "plots/Regularization_Regular_dropout_data.png");

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

	torch::Tensor data_train;
	torch::Tensor data_test;
	torch::Tensor labels_train;
	torch::Tensor labels_test;

	std::pair<int, int> train_test_sizes = train_test_split(0.8f, data, labels, data_train, labels_train, data_test, labels_test);

	auto train_dataset = DataLoader(data_train, labels_train).map(torch::data::transforms::Stack<>());

	int batchSize = 16;
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batchSize);

#pragma endregion

	const float dropoutRate = 0.2;
	std::shared_ptr<Net> net = std::make_shared<Net>(dropoutRate);

	float learningRate = 0.002f;
	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learningRate));

	int numEpochs = 1000;

	for (int i = 0; i < numEpochs; i++)
	{
		net->train();
		for (auto& batch : *train_loader)
		{
			torch::Tensor yhat = net->forward(batch.data);

			torch::Tensor loss = torch::binary_cross_entropy_with_logits(yhat, batch.target);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}
		// test model
		net->eval();
		if (i % 50 == 0)
		{
			{
				std::cout << std::setprecision(4) << "Epoch : " << setw(3) << i << "(" << numEpochs << ")" << " || ";
				torch::Tensor y_pred = net->forward(data_train);

				int numPreds = y_pred.size(0);
				int num_correct_preds = 0;
				for (int i = 0; i < numPreds; i++)
				{
					if ((int)(y_pred[i].item<float>() > 0) == labels_train[i].item<int>())
						num_correct_preds++;
				}

				std::cout << "Training Accuracy : " << setw(6) << (num_correct_preds * 100) / float(numPreds) << "%" << " | ";
			}
			{
				torch::Tensor y_pred = net->forward(data_test);

				int numPreds = y_pred.size(0);
				int num_correct_preds = 0;
				for (int i = 0; i < numPreds; i++)
				{
					if ((int)(y_pred[i].item<float>() > 0) == labels_test[i].item<int>())
						num_correct_preds++;
				}

				std::cout << "Test Accuracy : " << setw(6) << (num_correct_preds * 100) / float(numPreds) << "%" << std::endl;
			}
		}
	}


}

void Regularization::L2_regularization()
{
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

	torch::Tensor labels = torch::vstack({ torch::zeros({nPerClust, 1}), torch::ones({nPerClust, 1}) });

	torch::Tensor dataA = torch::transpose(torch::stack({ ax_t, ay_t }), 1, 0);
	torch::Tensor dataB = torch::transpose(torch::stack({ bx_t, by_t }), 1, 0);

	torch::Tensor data = torch::vstack({ dataA, dataB });

#pragma endregion

#pragma region creating and train the model

	torch::nn::Sequential ANNclassify(
		torch::nn::Linear(2, 1),
		torch::nn::Linear(1, 1)
	);

	float learningRate = 0.01f;
	float L2lambda = 0.01f; // Lambda values for L2 regularization
	torch::optim::SGD optimizer(ANNclassify->parameters(), torch::optim::SGDOptions(learningRate).weight_decay(L2lambda));

	int numepochs = 250;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		ANNclassify->train();
		torch::Tensor yhat = ANNclassify->forward(data);

		torch::Tensor loss = torch::binary_cross_entropy_with_logits(yhat, labels);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
#pragma endregion

#pragma region make predictions

	ANNclassify->eval();
	torch::Tensor predictions = ANNclassify->forward(data);

	int errors = 0;
	bool y = false;
	bool yHat = false;

	float threshold = 0;

	for (int i = 0; i < nPerClust * 2; i++) {

		y = false;
		yHat = false;

		if (labels[i].item<int>() == 1)
			y = true;
		if (predictions[i].item<float>() > threshold)
			yHat = true;

		if (y != yHat) {
			errors++;
		}
	}
	std::cout << "Accuracy : " << 100 * (((nPerClust * 2) - errors) / float(nPerClust * 2)) << "%" << std::endl;

#pragma endregion
}

void Regularization::AllCalls()
{
	train_and_eval_modes();
	Dropout_regularization();
	Regular_dropout();
	L2_regularization();
}

#pragma endregion
