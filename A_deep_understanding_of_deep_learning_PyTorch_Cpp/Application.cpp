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

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root)
{
	std::vector<std::vector<std::string>> data;
	std::vector<std::string> colNames;

	read_csv(root, data, colNames);

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

	return { X, y };
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
	explicit IrisDataSet(const std::string& root)
	{
		auto data = read_data(root);
		X_ = std::move(data.first);
		y_ = std::move(data.second);
	}
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

	int num_data_points = read_csv("./datasets/iris.csv", X, y, colNames);

	std::vector<std::vector<float>> X_train;
	std::vector<std::vector<float>> X_test;
	std::vector<int> y_train;
	std::vector<int> y_test;

	float train_percentage = 0.8;

	auto train_test_sizes = train_test_split(X, y, X_train, X_test, y_train, y_test, train_percentage, num_data_points);

	std::cout << train_test_sizes.first << std::endl;
	std::cout << train_test_sizes.second << std::endl;

	std::cout << y_train.size() << std::endl;
	std::cout << y_test.size() << std::endl;

	
	for (int i : y_train)
	{
		std::cout << i <<" ";
	}
	std::cout << std::endl;

	for (int i : y_test)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
	

	/*
	std::string root = "./datasets/iris.csv";
	int batchSize = 4;

	auto dataset = IrisDataSet(root).map(torch::data::transforms::Stack<>());;

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(dataset), batchSize);

	for (auto& batch : *data_loader)
	{
		auto X = batch.data;
		auto y = batch.target;

		std::cout << X << std::endl;
		std::cout << y << std::endl;
		std::cout << std::endl;

	}
	*/
	return 0;
}

