#include "pch.h"

constexpr int kTrainSize = 80;
constexpr int kTestSize = 20;

constexpr int kRows = 300;
constexpr int kCols = 300;

void randomFlips(cv::Mat& img)
{
	// Random flips in X and Y axises
	if (torch::randint(2, 1).item<int>() == 1) cv::flip(img, img, 0);
	if (torch::randint(2, 1).item<int>() == 1) cv::flip(img, img, 1);
}

torch::Tensor CVtoTensor(cv::Mat img, bool randFlips)
{
	cv::resize(img, img, cv::Size(kRows, kCols), 0, 0, cv::INTER_LINEAR);

	if (randFlips) randomFlips(img);

	auto img_tensor = torch::from_blob(img.data, { kRows, kCols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 }).toType(torch::kFloat).div_(255);
	return img_tensor;
}

std::pair<std::vector<std::string>, torch::Tensor> read_data(const std::string& root, bool train)
{
	int i = 0;
	std::string ext("/*.jpg");
	const auto num_samples = train ? kTrainSize : kTestSize;
	const auto folder = train ? root + "/train" : root + "/test";
	auto targets = torch::empty(num_samples, torch::kInt64);
	std::vector<std::string> images(num_samples);

	std::string cat_folder = folder + "/cats";
	std::string dog_folder = folder + "/dogs";
	std::vector<std::string> folders = { cat_folder, dog_folder };

	std::cout << "Loading image paths from " << folder << std::endl;

	int64_t label = 0;
	for (auto& f : folders)
	{
		std::vector<std::string> files;
		cv::glob(f + ext, files, false);

		for (std::string& p : files)
		{
			images[i] = p;
			targets[i] = torch::tensor(label, torch::kInt64);
			i++;
		}
		label++;
	}
	return { images, targets };
}

struct CatGog : torch::data::datasets::Dataset<CatGog>
{
public:
	enum class Mode { kTrain, kTest };

	explicit CatGog(const std::string& root, Mode mode, bool randFlips) :mode_(mode), randFlips_(randFlips)
	{
		auto data = read_data(root, (mode == Mode::kTrain));
		images_ = std::move(data.first);
		targets_ = std::move(data.second);
	}
	torch::data::Example<> get(size_t index) override
	{
		cv::Mat img = cv::imread(images_[index]);
		auto img_tensor = CVtoTensor(img, randFlips_);
		return { img_tensor, targets_[index] };
	}
	torch::optional<size_t> size() const override
	{
		return images_.size();
	}
	bool is_train() const noexcept
	{
		return (mode_ == Mode::kTrain);
	}
	const std::vector<std::string>& images() const
	{
		return images_;
	}
	const torch::Tensor& targets() const
	{
		return targets_;
	}
private:
	std::vector<std::string> images_;
	torch::Tensor targets_;
	Mode mode_;
	bool randFlips_;
};

cv::Mat TensorToCV(torch::Tensor x)
{
	x = x.permute({ 1, 2, 0 });
	x = x.mul(255).clamp(0, 255).to(torch::kByte);
	x = x.contiguous();

	int height = x.size(0);
	int width = x.size(1);

	cv::Mat output(cv::Size(width, height), CV_8UC3);
	std::memcpy((void*)output.data, x.data_ptr(), sizeof(torch::kU8) * x.numel());
	return output.clone();
}

void Run()
{
	const std::string root = "./dataset";
	int batchSize = 4;

	auto train_dataset = CatGog(root, CatGog::Mode::kTest, true)
		.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batchSize);

	auto test_dataset = CatGog(root, CatGog::Mode::kTest, false)
		.map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(test_dataset), kTestSize);

	int waitTime = 100;

	for (auto& batch : *train_loader)
	{
		auto img = batch.data;
		auto labels = batch.target;

		for (int i = 0; i < img.size(0); i++)
		{
			auto out = TensorToCV(img[i]);
			cv::imshow("win", out);
			std::cout << labels[i].item<int>() << " ";
			int k = cv::waitKey(waitTime);
		}
		std::cout << std::endl;
	}

	for (auto& batch : *test_loader)
	{
		auto img = batch.data;
		auto labels = batch.target;

		for (int i = 0; i < img.size(0); i++)
		{
			auto out = TensorToCV(img[i]);
			cv::imshow("win", out);
			std::cout << labels[i].item<int>() << " ";
			int k = cv::waitKey(waitTime);
		}
		std::cout << std::endl;
	}
}

int main()
{
	Run();
	return 0;
}