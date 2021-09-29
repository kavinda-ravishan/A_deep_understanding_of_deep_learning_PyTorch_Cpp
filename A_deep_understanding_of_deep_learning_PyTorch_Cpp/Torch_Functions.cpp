#include "pch.h"

namespace Math_numpy_PyTorch {

	//using namespace Math_numpy_PyTorch;
	//namespace t = Math_numpy_PyTorch;

	void Tensor() {

		torch::Tensor s = torch::tensor(3, torch::kInt);
		torch::Tensor v = torch::tensor(
			{ 1, 2, 3 },
			torch::kInt
		);
		torch::Tensor M = torch::tensor(
			{
				{1, 2, 3},
				{4, 5, 6}
			},
			torch::kInt
		);

		for (int i = 0; i < 6; i++) {

			std::cout << *(M.data_ptr<int>() + i) << std::endl;
		}


		std::cout << s << std::endl;
		std::cout << s.item<int>() << std::endl;

		std::cout << v << std::endl;
		std::cout << v.size(0) << std::endl;

		std::cout << M << std::endl;
		std::cout << M.size(0) << std::endl;
		std::cout << M.size(1) << std::endl;

	}

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

			std::cout << std::setw(11) << expLogx[i].item<float>() << " - " << logExpx[i].item<float>() << std::endl;
		}
	}

	void Entropyand_cross_entropy() {

		torch::Tensor p = torch::tensor({ 1, 0 }, torch::kFloat64);
		torch::Tensor q = torch::tensor({ 0.25, 0.75 }, torch::kFloat64);
		torch::Tensor H = torch::zeros({ 1 });

		for (int i = 0; i < p.sizes()[0]; i++) {

			H -= p[i] * torch::log(q[i]);
		}
		std::cout << "Cross-entropy : " << H << std::endl;


		H = -(p[0] * torch::log(q[0]) + p[1] * torch::log(q[1]));
		std::cout << "Cross-entropy : " << H << std::endl;

		H = torch::binary_cross_entropy(q, p);
		std::cout << "Cross-entropy : " << H << std::endl;

	}

	void Min_maxand_argmin_argmax() {

		torch::Tensor v = torch::tensor({ 1, 20, 2, -3 }, torch::kInt);

		torch::Tensor minIndex = torch::argmin(v);
		torch::Tensor maxIndex = torch::argmax(v);

		std::cout << v << std::endl;

		std::cout << "Min index : " << minIndex.item<int>() << std::endl;
		std::cout << "Max index : " << maxIndex.item<int>() << std::endl;

		std::cout << "Min : " << v[minIndex].item<int>() << std::endl;
		std::cout << "Max : " << v[maxIndex].item<int>() << std::endl;
		std::cout << std::endl;

		torch::Tensor M = torch::tensor(
			{
				{0, 1, 10},
				{20, 8, 5}
			},
			torch::kInt
		);

		std::cout << M << std::endl;

		torch::Tensor min1 = torch::min(M);
		std::tuple<torch::Tensor, torch::Tensor> min2 = torch::min(M, 0);
		std::tuple<torch::Tensor, torch::Tensor> min3 = torch::min(M, 1);

		std::cout << min1 << std::endl;
		std::cout << std::get<0>(min2) << std::endl;
		std::cout << std::get<1>(min2) << std::endl;
		std::cout << std::get<0>(min3) << std::endl;
		std::cout << std::get<1>(min3) << std::endl;

		std::cout << torch::argmin(M) << std::endl;
		std::cout << torch::argmin(M, 0) << std::endl;
		std::cout << torch::argmin(M, 1) << std::endl;

	}

	void Mean_variance() {

		torch::Tensor x = torch::tensor(
			{ 1, 2, 4, 6, 5, 4, 0 },
			torch::kFloat64
		);

		torch::Tensor mean = torch::sum(x) / x.size(0);
		torch::Tensor variance = torch::sum(torch::square(x - mean)) / (x.size(0) - torch::tensor(1, torch::kFloat64));
		torch::Tensor stdev = torch::sqrt(variance);

		std::cout << "Mean : " << torch::mean(x).item<float>() << ", " << mean.item<float>() << std::endl;
		std::cout << "Variance : " << torch::square(torch::std(x)).item<float>() << ", " << variance.item<float>() << std::endl;
		std::cout << "Standard deviation : " << torch::std(x).item<float>() << ", " << stdev.item<float>() << std::endl;
	}

	torch::Tensor Random_choice(torch::Tensor a, int size) {

		int n = a.size(0);
		int c;
		torch::Tensor sample = torch::zeros(size, a.dtype());

		for (int i = 0; i < size; i++) {

			c = torch::randint(n, 1).item<int>();
			sample[i] = a[c];
		}

		return sample;
	}

	void Random_samplingand_sampling_variability() {

		std::vector<int> val = { 1, 2, 3, 4, -1, -5, 0, 1, 4, 9, 6, -9, 6, -4, 2, 1 };
		int sampleSize = 5;
		int nExers = 1000;

		torch::Tensor x = torch::tensor(
			val,
			torch::kFloat64
		);

		torch::Tensor popMean = torch::mean(x);
		torch::Tensor sampleMeans = torch::zeros(nExers);
		torch::Tensor sample = torch::zeros(sampleSize);

		for (int i = 0; i < nExers; i++) {

			sample = Random_choice(x, sampleSize);
			sampleMeans[i] = torch::mean(sample);
		}

		std::cout << popMean << std::endl;
		std::cout << torch::mean(sampleMeans) << std::endl;

	}

	void Reproducible_randomness_seeding() {

		std::cout << torch::randn({ 1, 5 }) << std::endl;

		torch::manual_seed(17);

		torch::Tensor a = torch::tensor({ {-1.4135, 0.2336, 0.0340, 0.3499, -0.0145} });
		torch::Tensor b = torch::tensor({ {-0.6124, -1.1835, -1.4831, 1.8004, 0.0096} });

		std::cout << a << std::endl;
		std::cout << torch::randn({ 1, 5 }) << std::endl;
		std::cout << b << std::endl;
		std::cout << torch::randn({ 1, 5 }) << std::endl;
	}

	void AllCalles() {
		Tensor();
		Vector_and_matrix_transpose();
		The_dot_product();
		Matrix_multiplication();
		Softmax();
		Logarithms();
		Entropyand_cross_entropy();
		Min_maxand_argmin_argmax();
		Mean_variance();
		Random_samplingand_sampling_variability();
		Reproducible_randomness_seeding();
	}
}

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
