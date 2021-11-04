#include "pch.h"

constexpr int kTrainSize = 80;
constexpr int kTestSize = 20;

constexpr int kRows = 300;
constexpr int kCols = 300;

torch::Tensor CVtoTensor(cv::Mat img)
{
	cv::resize(img, img, cv::Size(kRows, kCols), 0, 0, cv::INTER_LINEAR);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	auto img_tensor = torch::from_blob(img.data, { kRows, kCols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).toType(torch::kFloat).div_(255);
	return img_tensor;
}

bool ListFiles(std::wstring path, std::wstring mask, std::vector<std::wstring>& files) {
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	std::wstring spec;
	std::stack<std::wstring> directories;

	directories.push(path);
	files.clear();

	while (!directories.empty()) {
		path = directories.top();
		spec = path + L"\\" + mask;
		directories.pop();

		hFind = FindFirstFile(spec.c_str(), &ffd);
		if (hFind == INVALID_HANDLE_VALUE) {
			return false;
		}

		do {
			if (wcscmp(ffd.cFileName, L".") != 0 &&
				wcscmp(ffd.cFileName, L"..") != 0) {
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					directories.push(path + L"\\" + ffd.cFileName);
				}
				else {
					files.push_back(path + L"\\" + ffd.cFileName);
				}
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES) {
			FindClose(hFind);
			return false;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
	}

	return true;
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train)
{
	int i = 0;
	std::wstring ext(L"*.jpg");
	const auto num_samples = train ? kTrainSize : kTestSize;
	const auto folder = train ? root + "/train" : root + "/test";
	auto targets = torch::empty(num_samples, torch::kInt64);
	auto images = torch::empty({ num_samples, 3, kRows, kCols }, torch::kFloat);

	std::string cat_folder = folder + "/cats";
	std::string dog_folder = folder + "/dogs";
	std::vector<std::string> folders = { cat_folder, dog_folder };

	std::vector<std::wstring> files;

	int64_t label = 0;
	for (auto& f : folders)
	{
		std::wstring path(f.begin(), f.end());
		if (ListFiles(path, ext, files)) {
			for (std::vector<std::wstring>::iterator it = files.begin(); it != files.end(); ++it)
			{
				std::cout << label << " : " << i << std::endl;

				std::wstring wstr = it->c_str();

				std::string str(wstr.begin(), wstr.end());
				cv::Mat img = cv::imread(str);
				auto img_tensor = CVtoTensor(img);
				images[i] = img_tensor;
				targets[i] = torch::tensor(label, torch::kInt64);
				i++;
			}
		}
		label++;
	}

	return { images, targets };
}

struct CatGog: torch::data::datasets::Dataset<CatGog>
{
public:
	enum Mode { kTrain, kTest };

	explicit CatGog(const std::string& root, Mode mode = Mode::kTrain) :mode_(mode)
	{
		auto data = read_data(root, (mode == Mode::kTrain));
		images_ = std::move(data.first);
		targets_ = std::move(data.second);
	}
	torch::data::Example<> get(size_t index) override
	{
		return { images_[index], targets_[index] };
	}
	torch::optional<size_t> size() const override
	{
		return images_.size(0);
	}
	bool is_train() const noexcept
	{
		return (mode_ == Mode::kTrain);
	}
	const torch::Tensor& images() const
	{
		return images_;
	}
	const torch::Tensor& targets() const
	{
		return targets_;
	}
private:
	torch::Tensor images_;
	torch::Tensor targets_;
	Mode mode_;
};

cv::Mat TensorToCV(torch::Tensor x)
{
	x = x.permute({ 1, 2, 0 });
	x = x.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
	x = x.contiguous();

	int height = x.size(0);
	int width = x.size(1);

	cv::Mat output(cv::Size(width, height), CV_8UC3);
	std::memcpy((void*)output.data, x.data_ptr(), sizeof(torch::kU8) * x.numel());
	return output.clone();
}

int main(int argc, char** args) 
{
	const std::string root = "./dataset";
	/*
	bool train = true;
	std::pair<torch::Tensor, torch::Tensor> data = read_data(root, train);
	*/

	auto train_dataset = CatGog(root)
		.map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }))
		.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), 4
		);

	auto test_dataset = CatGog(root, CatGog::Mode::kTest)
		.map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }))
		.map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(test_dataset), 4
		);

	for (auto& batch : *train_loader)
	{
		auto img = batch.data;
		auto labels = batch.target;

		auto out = TensorToCV(img[0]);
		cv::imshow("win", out);
		int k = cv::waitKey(0);
		break;
	}


	return 0;
}





