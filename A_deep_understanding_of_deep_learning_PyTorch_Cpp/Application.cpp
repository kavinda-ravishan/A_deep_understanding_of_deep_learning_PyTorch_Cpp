#include "pch.h"
#include "torch_functions.h"

void ALL() {
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();
}

std::vector<std::string> split(const std::string& s, char delimiter) {

	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

int read_csv(
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
int read_csv(
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

std::pair<int, int> train_test_split(
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

struct IrisDataSet : torch::data::datasets::Dataset<IrisDataSet>
{
public:
	explicit IrisDataSet(torch::Tensor& X, torch::Tensor& y):X_(X), y_(y){}
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

int main(int argc, char** args) {
	
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

	
	int batchSize = 10;

	auto train_dataset = IrisDataSet(X_train_t, y_train_t).map(torch::data::transforms::Stack<>());

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

	int numepochs = 100;

	for (int i = 0; i < numepochs; i++) {

		for (auto& batch : *train_loader)
		{
			auto X = batch.data;
			auto y = batch.target;

			torch::Tensor yhat = ANNclassify->forward(X);

			torch::Tensor loss = torch::nn::functional::cross_entropy(yhat, y);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		torch::Tensor pred = torch::argmax(ANNclassify->forward(X_test_t), 1);

		int numPreds = pred.size(0);
		int num_correct_preds = 0;
		for (int i = 0; i < numPreds; i++)
		{
			if (pred[i].item<int>() == y_test_t[i].item<int>()) num_correct_preds++;
		}

		std::cout << "Accuracy : " << (num_correct_preds * 100) / float(numPreds) << "%" << std::endl;
	}

	
	return 0;
}

