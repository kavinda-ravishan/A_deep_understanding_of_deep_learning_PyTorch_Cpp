#include "pch.h"

namespace Math_numpy_PyTorch {

	void Vector_and_matrix_transpose() {

		// Vector and matrix transpose

		torch::Tensor tv = torch::randint(1, 100, { 1,3 });
		torch::Tensor tM = torch::randint(1, 100, { 2,3 });
		torch::Tensor tharray = torch::tensor({ 1, 2, 3, 4, 5 }, { torch::kFloat64 });

		std::cout << tv << std::endl;
		std::cout << tv.t() << std::endl;

		std::cout << tM << std::endl;
		std::cout << tM.t() << std::endl;

		for (int i = 0; i < 3; i++) {

			std::cout << tv[0][i].item<float>() << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < 3; i++) {
				std::cout << tM[j][i].item<float>() << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		for (int i = 0; i < 3; i++) {

			tv[0][i] = 1;
		}
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < 3; i++) {
				tM[j][i] = 1;
			}
		}

		for (int i = 0; i < 3; i++) {

			std::cout << tv[0][i].item<float>() << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < 3; i++) {
				std::cout << tM[j][i].item<float>() << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;


		for (int i = 0; i < 5; i++) {

			std::cout << tharray[i].item<float>() << " ";
		}
		std::cout << std::endl;
	}

	void The_dot_product() {

		torch::Tensor tv1 = torch::tensor({ 1, 2, 3, 4 }, { torch::kInt64 });
		torch::Tensor tv2 = torch::tensor({ 0, 1, 0, -1 }, { torch::kInt64 });

		std::cout << torch::dot(tv1, tv2).item<int>() << std::endl;
		std::cout << torch::sum(tv1 * tv2).item<int>() << std::endl;
	}

	void Matrix_multiplication() {

		torch::Tensor A = torch::randn({ 3,4 });
		torch::Tensor B = torch::randn({ 4,5 });
		torch::Tensor C = torch::randn({ 4,5 });

		std::cout << torch::matmul(A, B) << std::endl;
		std::cout << torch::matmul(B.t(), C) << std::endl;
	}

	void Softmax() {

		torch::Tensor z = torch::tensor(
			{ {1}, {2}, {3} },
			{ torch::kFloat64 }
		);
	
		torch::Tensor sigma1 = torch::softmax(z, 0);
		std::cout << sigma1 << std::endl;

		torch::Tensor num = torch::exp(z);
		torch::Tensor denVal = torch::sum(torch::exp(z));
		torch::Tensor den = torch::full({ 3, 1 }, denVal.item<float>());
		torch::Tensor sigma2 = num / den;

		std::cout << sigma2 << std::endl;
	}

	void Logarithms() {

		int size = 5;

		torch::Tensor x = torch::linspace(0.0001, 1, size);

		torch::Tensor logx = torch::log(x);
		torch::Tensor expx = torch::exp(x);

		std::cout << "Log(x) :\n" << logx << std::endl;
		std::cout << "Exp(x) :\n" << expx << std::endl;

		torch::Tensor expLogx = torch::exp(logx);
		torch::Tensor logExpx = torch::log(expx);

		std::cout << "Exp(Log(x)) - Log(Exp(x))" << std::endl;

		for (int i = 0; i < size; i++) {

			std::cout<< std::setw(11) << expLogx[i].item<float>()<<" - "<< logExpx[i].item<float>() << std::endl;
		}
	}
}

using namespace Math_numpy_PyTorch;

int main(int argc, char** args) {

/*
	Vector_and_matrix_transpose();
	The_dot_product();
	Matrix_multiplication();
	Softmax();
	Logarithms();
*/
	Logarithms();

	return 0;
}