/*
#pragma once

#include "torch/torch.h"

struct NetImpl : torch::nn::Module {

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, out{ nullptr };

	NetImpl(int fc1_dims, int fc2_dims)
		:fc1(fc1_dims, fc1_dims), fc2(fc1_dims, fc2_dims), out(fc2_dims, 1)
	{
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("out", out);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(fc1(x));
		x = torch::relu(fc2(x));
		x = torch::relu(out(x));

		return x;
	}
};

TORCH_MODULE(Net);

#include "Network.h"

#include "torch/script.h" // Transferring_Model_from_Python

void Building_a_Neural_Network() {

	Net network(50, 10);
	std::cout << network << std::endl;

	torch::Tensor x, output;

	x = torch::randn({ 2, 50 });

	output = network->forward(x);
	std::cout << output << std::endl;
}

void Transferring_Model_from_Python() {

	torch::jit::script::Module net = torch::jit::load("./python/net.pt");

	torch::Tensor x = torch::randn({ 1, 100 });

	std::vector<torch::jit::IValue> input;

	input.push_back(x);

	torch::jit::IValue out = net.forward(input);

	std::cout << x << std::endl;
	std::cout << std::endl;
	std::cout << out << std::endl;
	std::cout << std::endl;
	std::cout << typeid(out).name() << std::endl;
}
*/