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
