#include "pch.h"

namespace Gradient_Descent {

	torch::Tensor fx(torch::Tensor& x) {
		return (3 * x.pow(2)) - (3 * x) + 4;
	}

	torch::Tensor deriv(torch::Tensor& x) {
		return (6 * x) - 3;
	}

	void Gradient_descent_in_1D(torch::Tensor(*deriv)(torch::Tensor& x)) {

		int x_len = 2001;
		torch::Tensor x = torch::linspace(-2, 2, x_len);
		torch::Tensor localMin = x[torch::randint(0, x_len, 1).item<int>()];

		std::cout << localMin << std::endl;

		int training_epochs = 100;
		torch::Tensor learning_rate = torch::tensor(0.01, torch::kFloat64);
		torch::Tensor grad = torch::tensor(0, torch::kFloat64);

		for (int i = 0; i < training_epochs; i++) {

			grad = deriv(localMin);
			localMin = localMin - (learning_rate * grad);
		}

		std::cout << localMin << std::endl;
	}
}

using namespace Gradient_Descent;

int main(int argc, char** args) {

	Gradient_descent_in_1D(deriv);

	return 0;
}