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

int main(int argc, char** args) {

	Building_a_Neural_Network();
	Transferring_Model_from_Python();
	
	return 0;
}

