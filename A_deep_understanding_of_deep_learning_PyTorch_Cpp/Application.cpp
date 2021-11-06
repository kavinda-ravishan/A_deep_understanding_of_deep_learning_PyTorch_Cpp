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
	
	//auto data = read_data("./datasets/iris.csv");
	//auto randIndex = torch::randperm(data.first.size(0));

	std::vector<std::vector<float>> X;
	std::vector<int> y;
	std::vector<std::string> colNames;

	int size = read_csv("./datasets/iris.csv", X, y, colNames);

	std::cout << colNames << std::endl;

	
	for (float i : X[0])
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
	for (float i : X[1])
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
	for (float i : X[2])
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
	for (float i : X[3])
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;

	for (float i : y)
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

