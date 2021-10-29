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
