#include "pch.h"
#include "torch_functions.h"

void ALL() {
	/*
	*/
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();
}


int main(int argc, char** args) {

	ALL();

	return 0;
}

