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

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train)
{
	int i = 0;
	std::string ext("/*.jpg");
	const auto num_samples = train ? kTrainSize : kTestSize;
	const auto folder = train ? root + "/train" : root + "/test";
	auto targets = torch::empty(num_samples, torch::kInt64);
	auto images = torch::empty({ num_samples, 3, kRows, kCols }, torch::kFloat);

	std::string cat_folder = folder + "/cats";
	std::string dog_folder = folder + "/dogs";
	std::vector<std::string> folders = { cat_folder, dog_folder };

	std::cout << "Loading images from "<< folder << std::endl;

	int64_t label = 0;
	for (auto& f : folders)
	{
		std::vector<std::string> files;
		cv::glob(f + ext, files, false);

		for (std::string& p : files)
		{
			std::cout << label << " : " << i << std::endl;

			cv::Mat img = cv::imread(p);
			auto img_tensor = CVtoTensor(img);
			images[i] = img_tensor;
			targets[i] = torch::tensor(label, torch::kInt64);

			i++;
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

	auto train_dataset = CatGog(root)
		.map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }))
		.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), 4);

	auto test_dataset = CatGog(root, CatGog::Mode::kTest)
		.map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }))
		.map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(test_dataset), kTestSize);

	for (auto& batch : *train_loader)
	{
		auto img = batch.data;
		auto labels = batch.target;

		for (int i = 0; i < img.size(0); i++)
		{
			auto out = TensorToCV(img[i]);
			cv::imshow("win", out);
			int k = cv::waitKey(10);
		}
	}

	for (auto& batch : *test_loader)
	{
		auto img = batch.data;
		auto labels = batch.target;

		std::cout << img.sizes() << std::endl;

		for (int i = 0; i < img.size(0); i++)
		{
			auto out = TensorToCV(img[i]);
			cv::imshow("win", out);
			int k = cv::waitKey(10);
		}
	}

	return 0;
}





