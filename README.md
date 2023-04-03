This is the C++ implementation of the improved rule induction algorithm in our IST paper. To run it, make the format of your dataset file(s) like `train.arff` and `test.arff` (the sample here is one fold of project `JDT` in dataset `AEEEM`). Then, compile the source file `ImbalanceRuleInduction.cpp`, and run it with the command line for specifying input and output, as well as some hyper-parameters. 

Note that we run our experiment on a windows PC with Microsoft Visual Studio 2019 (the corresponding MSVC complier). We do not guarantee there are no issues when it is on Linux platform with other complier. Nevertheless, it should be complied with C++11 standard, and to make sure it can run fastly, the option -O2 should be turned on.

Please cite our paper if applicable: 
`Gao et al. Dealing with imbalanced data for interpretable defect prediction. Information and Software Technology, 2022, 151. doi: 10.1016/j.infsof.2022.107016.`
