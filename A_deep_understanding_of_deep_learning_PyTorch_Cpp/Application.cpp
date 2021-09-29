#include "pch.h"

int main(int argc, char** args) {

#pragma region Create data

	int N = 30;

	torch::Tensor x = torch::randn({ N,1 }, torch::kFloat32);
	torch::Tensor y = x + torch::randn({ N,1 }, torch::kFloat32)/2;

#pragma endregion

	torch::nn::Sequential ANNReg(
		torch::nn::Linear(1,1),
		torch::nn::ReLU(),
		torch::nn::Linear(1,1)
	);

	float learningRate = 0.05;
	torch::optim::SGD optimizer(ANNReg->parameters(), learningRate);

	int numepochs = 500;
	torch::Tensor losses = torch::zeros(numepochs, torch::kFloat32);

	for (int i = 0; i < numepochs; i++) {
		torch::Tensor yhat = ANNReg->forward(x);

		torch::Tensor loss = torch::mse_loss(yhat, y);
		losses[i] = loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
	
	torch::Tensor predictions = ANNReg->forward(x);

	std::cout << x << std::endl;
	std::cout << y << std::endl;
	std::cout << predictions << std::endl;

	return 0;
}