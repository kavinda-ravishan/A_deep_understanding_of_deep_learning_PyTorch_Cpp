#pragma once
#include "pch.h"

namespace Math_numpy_PyTorch {

	//using namespace Math_numpy_PyTorch;
	//namespace t = Math_numpy_PyTorch;

	void Tensor();

	void Vector_and_matrix_transpose();

	void The_dot_product();

	void Matrix_multiplication();

	void Softmax();

	void Logarithms();

	void Entropyand_cross_entropy();

	void Min_maxand_argmin_argmax();

	void Mean_variance();

	torch::Tensor Random_choice(torch::Tensor a, int size);

	void Random_samplingand_sampling_variability();

	void Reproducible_randomness_seeding();

	void AllCalls();
}

namespace Gradient_Descent {

	torch::Tensor fx(torch::Tensor& x);

	torch::Tensor deriv(torch::Tensor& x);

	void Gradient_descent_in_1D(torch::Tensor(*deriv)(torch::Tensor& x));

	void AllCalls();
}

namespace ANNs {

	void plot_data(
		std::vector<double>& ax,
		std::vector<double>& ay,
		std::vector<double>& bx,
		std::vector<double>& by,
		std::string path
	);

	void plot_data_err(
		std::vector<double>& ax,
		std::vector<double>& ay,
		std::vector<double>& bx,
		std::vector<double>& by,
		std::vector<double>& errx,
		std::vector<double>& erry,
		std::string path
	);

	void plot_losses(torch::Tensor losses_t, int numepochs, std::string path);

	std::vector<std::string> split(const std::string& s, char delimiter);

	void read_csv(
		std::string path,
		std::vector<std::vector<std::string>>& data,
		std::vector<std::string>& colNames
	);

	void ANN_regression();

	void ANN_classification();

	void Multilayer_ANN_classification();

	void ANN_iris_dataset();

	void Number_of_parameters();

	class ANNclassifyClass :public torch::nn::Module {

	private:
		torch::nn::Linear input{ nullptr }, output{ nullptr };

	public:
		ANNclassifyClass();
		torch::Tensor forward(torch::Tensor x);
	};

	void ANN_class_classification();

	class ANN_ModuleDict : public torch::nn::Module {

	private:
		torch::nn::ModuleDict ann_dict{ nullptr };
		std::vector<std::string> layersNames;

	public:
		ANN_ModuleDict(int nUnits, int nLayers);

		torch::Tensor forward(torch::Tensor x);
	};

	void ANN_class_ModuleDict_classification();

	void AllCalls();
}

namespace Overfittingand_cross_validation {

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

	void iris_dataset_dataloader_barch_train();

	void ALLCalls();
}