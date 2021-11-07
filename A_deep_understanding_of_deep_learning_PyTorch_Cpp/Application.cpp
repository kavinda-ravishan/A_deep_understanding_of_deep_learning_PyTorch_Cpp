#include "pch.h"
#include "torch_functions.h"

void ALL() {
	Math_numpy_PyTorch::AllCalls();
	Gradient_Descent::AllCalls();
	ANNs::AllCalls();
	Overfittingand_cross_validation::ALLCalls();
}

int main(int argc, char** args) {
	
	Overfittingand_cross_validation::ALLCalls();
	
	return 0;
}

