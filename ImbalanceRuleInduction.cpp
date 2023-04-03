#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <functional>
#include <set>
#include <algorithm>
#include <list>
#include <iomanip> 
#include <sstream>

#define DEBUG

namespace Out {
	void fatalError(std::string message) {
		std::cout << "Fatal: " << message << std::endl;
		exit(-1);
	}

	void printSet(std::unordered_set<size_t> s) {
		std::cout << "{";
		for (auto val : s)
			std::cout << val << ", ";
		std::cout << "}" << std::endl;
	}
}

namespace Utils {

	static const char Space = ' ';
	static const char EndChar = '\0';
	static const char Comma = ',';

	double CorrectedConfidence(double minimumConfidence, double imbRate, double sigma) {
		return minimumConfidence * exp(-1 * (imbRate - 0.5) * (imbRate - 0.5) / (2 * sigma * sigma));
	}

	double CorrectedConfidenceCosineh(double minimumConfidence, double imbRate, double sigma) {
		return minimumConfidence * cos(imbRate - 0.5) * sigma;
	}

	template <typename T>
	bool isSubset(const std::unordered_set<T>& a, const std::unordered_set<T>& b)
	{
		// return true if all members of a are also in b
		auto const is_in_b = [&b](auto const& x) { return b.find(x) != b.end(); };
		return std::all_of(a.begin(), a.end(), is_in_b);
	}

	double toDouble(std::string& s) {
		return atof(s.c_str());
	}

	std::string doubleToString(double d, const char* format) {
		char buf[32];
		snprintf(buf, 32, format, d);
		return std::string(buf);
	}

	std::string integerToString(int i) {
		char buf[16];
		snprintf(buf, 16, "%d", i);
		return std::string(buf);
	}

	std::string trim(std::string& s) {
		if (s.empty()) {
			return std::string(s);
		}
		std::string result;
		for (size_t i = 0; i < s.length(); i++) {
			if (s[i] != Utils::Space)
				result.push_back(s[i]);
		}
		return result;
	}

	// Implementation of python str.join().
	std::string join(std::string& delimiter, std::vector<std::string> v) {
		if (v.size() == 0) {
			return std::string("");
		}
		if (v.size() == 1) {
			return v[0];
		}
		std::string result;
		result = v[0];
		size_t _sizeToReserve = delimiter.length() * v.size() + 64;
		for (size_t i = 0; i < v.size(); i++) {
			_sizeToReserve += v[i].length();
		}
		result.reserve(_sizeToReserve);
		for (size_t i = 1; i < v.size(); i++) {
			result.append(delimiter);
			result.append(v[i]);
		}
		return result;
	}

	// Implementation of python str.split().
	const std::vector<std::string>& split(const std::string& s, const char delimiter, std::vector<std::string>& container) {
		size_t _pos = 0;
		size_t _lastPos = -1;
		while (true) {
			_pos = s.find(delimiter, _lastPos + 1);
			if (_pos == std::string::npos) {
				break;
			}
			container.push_back(s.substr(_lastPos + 1, _pos - _lastPos - 1));
			_lastPos = _pos;
		}
		container.push_back(s.substr(_lastPos + 1));
		return container;
	}
};

class Dataset {
public:

	class Instance {
	private:
		Dataset* m_DatasetPtr = nullptr;
		size_t m_InstanceOffset = 0;
		size_t m_InstanceId = -1;
	public:
		Instance(Dataset* dataset, size_t instanceId) {
#ifdef _DEBUG
			assert(dataset != nullptr);
#endif
			this->m_DatasetPtr = dataset;
			this->m_InstanceOffset = instanceId * dataset->Attributes().size();
			this->m_InstanceId = instanceId;
		}

		const std::string& GetLabel() const {
			return this->m_DatasetPtr->m_DataY[this->m_InstanceId];
		}

		double GetValueAt(size_t attributeId) const {
			if (attributeId >= this->m_DatasetPtr->Attributes().size()) {
				Out::fatalError("Attribute id out of range.");
				throw "Attribute id out of range";
			}
			return this->m_DatasetPtr->m_DataX[this->m_InstanceOffset + attributeId];
		}

		double GetValueAt(const std::string& attributeName) const {
			auto _colIter = this->m_DatasetPtr->m_AttributeStringToIndex.find(attributeName);
			if (_colIter == this->m_DatasetPtr->m_AttributeStringToIndex.end()) {
				Out::fatalError("Key error. No such attribute named.");
			}
			return this->m_DatasetPtr->m_DataX[this->m_InstanceOffset + _colIter->second];
		}
	};

private:
	static const int IOError = 1;
	static const int ARFFError = 2;
	static const char CharSpace = ' ';

	int m_initStatus = 0;
	std::string m_RelationName;
	std::vector<std::string> m_Attributes;
	std::unordered_map<std::string, size_t> m_AttributeStringToIndex;
	double* m_DataX = nullptr;
	std::vector<std::string> m_DataY;
	size_t m_DataCount = 0;
	std::vector<std::string> m_Labels;

	inline void _boundaryCheck(size_t r, size_t c) {
		// Assume the column is always checked.
		if (r >= m_DataCount) {
			Out::fatalError("Row out of range.");
			return;
		}
#ifdef _DEBUG
		// For debug purpose do the boundary check again.
		if (c >= m_Attributes.size()) {
			Out::fatalError("Column out of range");
			return;
		}
#endif
	}

	double _ValueAt(size_t row, size_t col) {
		this->_boundaryCheck(row, col);
		return this->m_DataX[row * this->m_Attributes.size() + col];
	}

	void _SetValueAt(size_t row, size_t col, double val) {
		// This function is private, so no boundary check is necessary.
#ifdef _DEBUG
		// Do boundary check for debug purpose.
		this->_boundaryCheck(row, col);
#endif
		this->m_DataX[row * this->m_Attributes.size() + col] = val;
		return;
	}

public:

	Instance GetInstance(size_t id) {
		if (id >= this->m_DataCount) {
			Out::fatalError("Instance id out of range.");
			throw "Instance id out of range.";
		}
		return Instance(this, id);
	}

	double GetValueAt(size_t row, size_t col) {
		if (col >= m_Attributes.size()) {
			Out::fatalError("Column out of range");
			throw "Column out of range";
		}
		return this->_ValueAt(row, col);
	}

	double GetValueAt(size_t i, const std::string& columnName) {
		auto _colIter = this->m_AttributeStringToIndex.find(columnName);
		if (_colIter == this->m_AttributeStringToIndex.end()) {
			Out::fatalError("Column key error.");
		}
		return this->_ValueAt(i, _colIter->second);
	}

	std::string& y(size_t i) {
		if (i >= m_DataCount) {
			Out::fatalError("Row out of range");
			throw "Row out of Range";
		}
		return this->m_DataY[i];
	}

	const std::string& RelationName() const {
		return this->m_RelationName;
	}

	const std::vector<std::string>& Labels() const {
		return this->m_Labels;
	}

	const std::vector<std::string>& Attributes() const {
		return this->m_Attributes;
	}

	size_t InstanceCount() {
		return this->m_DataCount;
	}

	Dataset(const char* arffFilename) {
		int _currentLine = 0;
		std::ifstream _fs;
		size_t _attrId = 0;
		size_t _labelId = 0;
		bool _dataSectionStart = false;
		std::unordered_set<size_t> _numericalAttributes;
		size_t _labelSubscript = 0;

		_fs.open(arffFilename, std::ios::in);
		if (_fs.rdstate() != std::ios::goodbit) {
			this->m_initStatus = IOError;
			goto ERROR;
		}
		// Then iteratively reading the lines.
		while (!_fs.eof()) {
			_currentLine += 1;
			std::string _line;
			getline(_fs, _line);
			if (_line[0] == '@') {
				// It is the field of arff data.
				// The valid weka arff data should contain 3 field, which are @relation,
				// @attribute and @data. Thus, we use the [1:4] for determination.
				std::string _prefixAt4 = _line.substr(1, 4);
				std::vector<std::string> _splited;
				if (_prefixAt4 == "rela") {
					//@relation
					Utils::split(_line, Utils::Space, _splited);
					if (_splited.size() < 2) {
						this->m_initStatus = ARFFError;
						goto ERROR;
					}
					this->m_RelationName = _splited[1];
				}
				else if (_prefixAt4 == "attr") {
					//@attribute
					Utils::split(_line, Utils::Space, _splited);
					// The @attribute line must be "@attribute <attribute_name> <attribute_type>".
					// Here we only support numerical for attributes and nominal for labels.
					if (_splited.size() != 3) {
						this->m_initStatus = ARFFError;
						goto ERROR;
					}
					std::string _AttributeName = _splited[1];
					std::string _AttributeType = _splited[2];
					if (_AttributeType == "numeric") {
						// Note: _attrId is the id of attribute in arff file, which may not be included in the Dataset object.
						// m_Attributes is the list (std::vector) of attributes which is captured by the Dataset object
						this->m_AttributeStringToIndex.insert(std::make_pair(_AttributeName, m_Attributes.size()));
						this->m_Attributes.push_back(_AttributeName);
						_numericalAttributes.insert(_attrId);
					}
					else {
						if (_AttributeType[0] != '{' || _AttributeType[_AttributeType.length() - 1] != '}') {
							std::cerr << "Warning: Ignore unknown type of attribute " << _AttributeName << "." << std::endl;
						}
						if (_labelId > 1) {
							std::cerr << "Warning: More than one nominal attribute detected. Only the last will be left as label." << std::endl;
						}
						_labelId++;
						_labelSubscript = _attrId;
						// Assume the other are nominal attributes for label.
						Utils::split(_AttributeType.substr(1, _AttributeType.length() - 2),
							Utils::Comma,
							this->m_Labels);
						for (size_t i = 0; i < this->m_Labels.size(); i++) {
							this->m_Labels[i] = Utils::trim(this->m_Labels[i]);
						}
					}
					_attrId++;
				}
				else if (_prefixAt4 == "data") {
					//@data
					// Note that data section contains data till the eof.
					size_t _dataId = 0;
					std::vector<double*> _data;
					while (!_fs.eof()) {
						std::vector<std::string> _splited;
						size_t _dataAttrId = 0;
						double* _dataLine = (double*)malloc(sizeof(double) * this->m_Attributes.size());
						getline(_fs, _line);
						if (_line.length() == 0) {
							std::cerr << "Info: Ignore empty line #" << _dataId << "." << std::endl;
							continue;
						}
						Utils::split(_line, Utils::Comma, _splited);
						if (_splited.size() == 0) {
							std::cerr << "Warning: Ignore invalid data #" << _dataId << "." << std::endl;
							continue;
						}
						for (size_t j = 0; j < _splited.size(); j++) {
							if (_numericalAttributes.find(j) == _numericalAttributes.end()) {
								if (j == _labelSubscript) {
									this->m_DataY.push_back(_splited[j]);
								}
								else {
									std::cerr << "Warning: Ignore data of unknown type: " << _splited[j] << "." << std::endl;
								}
							}
							else {
								_dataLine[_dataAttrId++] = Utils::toDouble(_splited[j]);
							}
						}
						if (_dataAttrId != this->m_Attributes.size()) {
							std::cerr << "Warning: Ignore invalid data #" << _dataId << "." << std::endl;
							continue;
						}
						_data.push_back(_dataLine);
						_dataId++;
					}

					// Copy the data to this->m_Data.
					this->m_DataCount = _data.size();
					this->m_DataX = (double*)malloc(sizeof(double) * this->m_Attributes.size() * _data.size());
					std::cout << this->m_Attributes.size() << ", " << _data.size() << std::endl;
					for (size_t i = 0; i < _data.size(); i++) {
						for (size_t j = 0; j < this->m_Attributes.size(); j++) {
							this->_SetValueAt(i, j, _data[i][j]);
						}
					}

					// Collect the memory of _dataLine(s).
					while (!_data.empty()) {
						double*& _dataPtr = _data.back();
						free(_dataPtr);
						_dataPtr = nullptr;
						_data.pop_back();
					}
				}
			}

		}
	ERROR:
		// Error handler.
		if (_fs.is_open()) {
			_fs.close();
		}

		switch (this->m_initStatus) {
		case IOError:
			std::cerr << "Error: IOError during opening the arff file." << std::endl;
			break;
		case ARFFError:
			std::cerr << "Error: ARFF file corrupted." << std::endl;
			break;
		}
	}

	~Dataset() {
		if (this->m_DataX != nullptr) {
			delete this->m_DataX;
		}
	}
};

class ConfusionMatrix {
private:
	size_t m_Matrix[2][2] = { 0 };
	size_t m_InstanceCount = 0;
public:
	ConfusionMatrix(std::vector<std::string>& y_pred, std::vector<std::string>& y_test) {
		if (y_pred.size() != y_test.size()) {
			Out::fatalError("Length of y_pred and y_test must be the same for computation of confusion matrix.");
		}
		this->m_InstanceCount = y_pred.size();
		for (size_t i = 0; i < y_pred.size(); i++) {
			size_t _actualLabel = y_test[i] == "1" ? 1 : 0;
			size_t _predictedLabel = y_pred[i] == "1" ? 1 : 0;
			m_Matrix[_actualLabel][_predictedLabel]++;
		}
	}

	double Accuracy() {
		return ((double)(m_Matrix[1][1]) + (double)(m_Matrix[0][0])) / (double)(this->m_InstanceCount);
	}

	double Precision(size_t targetClass = 1) {
		if ((m_Matrix[targetClass][targetClass]) + (m_Matrix[1 - targetClass][targetClass]) == 0) {
			// Ill-defined precision.
			return 0;
		}
		return (double)(m_Matrix[targetClass][targetClass]) / ((double)(m_Matrix[targetClass][targetClass]) + (double)(m_Matrix[1 - targetClass][targetClass]));
	}

	double Recall(size_t targetClass = 1) {
		if (m_Matrix[targetClass][targetClass] + m_Matrix[targetClass][1 - targetClass] == 0) {
			// Ill-defined Recall
			return 0;
		}
		return (double)(m_Matrix[targetClass][targetClass]) / ((double)(m_Matrix[targetClass][targetClass]) + (double)(m_Matrix[targetClass][1 - targetClass]));
	}

	double F1Score(size_t targetClass = 1) {
		double p = Precision(targetClass);
		double r = Recall(targetClass);
		if (p == 0 && r == 0) {
			// Ill-defined f1.
			return 0.0;
		}
		return 2 * p * r / (p + r);
	}

	double FalsePositiveRate(size_t targetClass = 1) {
		if (m_Matrix[1 - targetClass][targetClass] + m_Matrix[1 - targetClass][1 - targetClass] == 0) {
			// Ill-defined fp rate.
			return 0.0;
		}
		return (double)(m_Matrix[1 - targetClass][targetClass]) / ((double)(m_Matrix[1 - targetClass][targetClass]) + (double)(m_Matrix[1 - targetClass][1 - targetClass]));
	}

	// Convert a ConfusionMatrix object to std::string like __str__ in python.
	std::string ToString() {
		// dummy method!
		std::string result;
		return result;
	}
};

class ROCurve {
public:
	ROCurve(std::vector<std::string>& yPred, std::vector<std::string>& yTest) {
		// Not implemented. Call sklearn.metrics.roc_auc_score() instead.
		Out::fatalError("Class ROCurve has not been implemented yet.");
	}
};

double RuleAUC(double Recall, double FPR) {
	// Since the score for all predictions are the same, the instances are at the same point in ROC curve.
	// Actual AUC score should be calculated with probabilities. Please call sklearn.metrics.roc_auc_score instead.
	return 0.5 * FPR * Recall + 0.5 * (Recall + 1) * (1 - FPR);
}


class ImbRuleInductionClassifier {
public:

#ifdef _DEBUG
	static size_t LabelStringToUnsigned(const std::string& label) {
		if (label == "1") {
			return 1;
		}
		return 0;
	}
#else
	static size_t LabelStringToUnsigned(const std::string& label) {
		if (label[0] == '1') {
			return 1;
		}
		return 0;
	}
#endif

	class Term {
	public:
		static const int OperatorEqual = 0;
		static const int OperatorLessEqual = 1;
		static const int OperatorGreaterEqual = 2;
		static const int OperatorLessThan = 3;
		static const int OperatorGreaterThan = 4;

	private:
		std::string m_AttributeName;
		int m_operatorType = -1;
		double m_Value;

	public:
		Term(std::string& attributeName, int operatorType, double value) {
			if (operatorType > 4 || operatorType < 0) {
				Out::fatalError("Invalid operator type for term.");
				throw "Invalid operator type for term.";
			}
			this->m_AttributeName = attributeName;
			this->m_operatorType = operatorType;
			this->m_Value = value;
		}

		Term(const Term& term) {
			// Copy constructor.
			this->m_AttributeName = term.m_AttributeName;
			this->m_operatorType = term.m_operatorType;
			this->m_Value = term.m_Value;
		}

		Term operator= (const Term& term) {
			this->m_AttributeName = term.m_AttributeName;
			this->m_operatorType = term.m_operatorType;
			this->m_Value = term.m_Value;
		}

		bool operator== (const Term& theOther) {
			return (this->m_AttributeName == theOther.m_AttributeName)
				&& (this->m_operatorType == theOther.m_operatorType)
				&& (this->m_Value == theOther.m_Value);
		}

		bool Satisfy(const Dataset::Instance& instance) {
			double val = instance.GetValueAt(this->m_AttributeName);
			switch (this->m_operatorType) {
			case OperatorEqual:
				return val == this->m_Value;
				break;
			case OperatorGreaterEqual:
				return val >= this->m_Value;
				break;
			case OperatorGreaterThan:
				return val > this->m_Value;
				break;
			case OperatorLessEqual:
				return val <= this->m_Value;
				break;
			case OperatorLessThan:
				return val < this->m_Value;
				break;
			}
			return false;
		}

		const std::string& Attribute() const {
			return this->m_AttributeName;
		}

		const int Operator() const {
			return this->m_operatorType;
		}

		const double Value() const {
			return this->m_Value;
		}

		std::string ToString() {
			std::string _valueStr = Utils::doubleToString(this->m_Value, "%.3f");
			std::string _result = "(";
			_result.append(this->m_AttributeName);
			switch (this->m_operatorType) {
			case OperatorEqual:
				_result.append(" = ");
				break;
			case OperatorGreaterThan:
				_result.append(" > ");
				break;
			case OperatorLessThan:
				_result.append(" < ");
				break;
			case OperatorGreaterEqual:
				_result.append(" >= ");
				break;
			case OperatorLessEqual:
				_result.append(" <= ");
				break;
			}
			std::ostringstream _ss;
			_ss << this->m_Value;
			_result.append(_ss.str()).append(")");
			return _result;
		}

	};

	// Class for a rule.
	class Rule {
		//
	protected:
		// Private fields.

		// Terms of the rule (antecedent).
		std::vector<Term> m_Terms;

		// The label of the rule (consequent).
		std::string m_Label;

		double m_Confidence;
		double m_Support;
		std::unordered_set<size_t> m_CoveredInstancesOfMajorClass;
		std::unordered_set<size_t> m_CoveredInstancesOfMinorClass;
		size_t m_TargetClass;
		Dataset* m_DatasetPtr = nullptr;
		bool m_DefaultRule = false;

	private:
		void _updateCoveredInstances() {
			Dataset& dataset = *(this->m_DatasetPtr);
			// First we need to get which label is the minor class. For rule induction algorithm, we start the rule induction 
			// from instances of minor class, and the label of default rule is always the major class.
			size_t _minorClass = this->isDefaultRule() ? 1 - this->m_TargetClass : this->m_TargetClass;
			// Update coverage.
			// Firstly we need to clear the coverage set.
			this->m_CoveredInstancesOfMajorClass.clear();
			this->m_CoveredInstancesOfMinorClass.clear();
			// Then we add the covered instances for each class.
			for (size_t i = 0; i < dataset.InstanceCount(); i++)
			{
				Dataset::Instance _instance = dataset.GetInstance(i);
				size_t instanceLabel = LabelStringToUnsigned(_instance.GetLabel());
				if (this->Cover(_instance)) {
					if (instanceLabel == _minorClass) {
						this->m_CoveredInstancesOfMinorClass.insert(i);
					}
					else {
						this->m_CoveredInstancesOfMajorClass.insert(i);
					}
				}
			}
		}

		void _calculateRuleMetrics(bool isDefaultRule = false) {
			if (isDefaultRule == false) {
				this->m_Support = (double)this->m_CoveredInstancesOfMinorClass.size() / (double)(this->m_DatasetPtr->InstanceCount());
				this->m_Confidence = (double)this->m_CoveredInstancesOfMinorClass.size() /
					((double)this->m_CoveredInstancesOfMinorClass.size() + (double)this->m_CoveredInstancesOfMajorClass.size());
			}
			else {
				this->m_Support = (double)this->m_CoveredInstancesOfMajorClass.size() / (double)(this->m_DatasetPtr->InstanceCount());
				this->m_Confidence = (double)this->m_CoveredInstancesOfMajorClass.size() /
					((double)this->m_CoveredInstancesOfMinorClass.size() + (double)this->m_CoveredInstancesOfMajorClass.size());
			}
		}
	public:

		bool isDefaultRule() {
			return this->m_DefaultRule;
		}

		Rule(const std::string& label, bool isDefaultRule, Dataset* datasetPtr) {
			this->m_Label = label;
			this->m_Terms.clear();
			this->m_DatasetPtr = datasetPtr;
			this->m_TargetClass = LabelStringToUnsigned(this->m_Label);
			this->_updateCoveredInstances();
			this->_calculateRuleMetrics();
		}

	private:
		// Copy the fields and data from an existing rule.
		void _copyFrom(const Rule& rule) {
			// Copy terms.
			this->m_Terms.clear();
			for (size_t i = 0; i < rule.m_Terms.size(); i++) {
				this->m_Terms.push_back(rule.m_Terms[i]);
			}
			// Copy fields.
			this->m_Label = rule.m_Label;
			this->m_DatasetPtr = rule.m_DatasetPtr;
			this->m_Support = rule.m_Support;
			this->m_Confidence = rule.m_Confidence;
			this->m_TargetClass = rule.m_TargetClass;
			this->m_DefaultRule = rule.m_DefaultRule;
			// Copy covered instances.
			this->m_CoveredInstancesOfMajorClass = std::unordered_set<size_t>(rule.m_CoveredInstancesOfMajorClass);
			this->m_CoveredInstancesOfMinorClass = std::unordered_set<size_t>(rule.m_CoveredInstancesOfMinorClass);
		}

	public: // Constructors
		// Construct a rule by given terms and label.
		Rule(const std::string& label, const std::vector<Term>* termsVectorPtr, Dataset* dataset) {
			this->m_Label = label;
			if (termsVectorPtr != nullptr) {
				for (size_t i = 0; i < termsVectorPtr->size(); i++) {
					this->m_Terms.push_back(termsVectorPtr->at(i));
				}
			}
			this->m_DatasetPtr = dataset;
			this->m_TargetClass = LabelStringToUnsigned(this->m_Label);
			this->_updateCoveredInstances();
			this->_calculateRuleMetrics();
		}
		// Construct a default rule.
		Rule(const std::string& label, const std::list<Rule>& minorClassRules, Dataset* dataset) {
			this->m_DefaultRule = true;
			this->m_Label = label;
			this->m_DatasetPtr = dataset;
			this->m_TargetClass = LabelStringToUnsigned(this->m_Label);
			this->_updateCoveredInstances();
			// Update minor.
			std::unordered_set<size_t> _minorClassRulesCovered;
			for (auto _rule : minorClassRules) {
				for (auto _ruleCoveredMinorInstances : _rule.MinorInstancesCovered()) {
					_minorClassRulesCovered.insert(_ruleCoveredMinorInstances);
				}
			}
			for (size_t _minorInstanceCovered : _minorClassRulesCovered) {
				this->m_CoveredInstancesOfMinorClass.erase(_minorInstanceCovered);
			}
			// Update major.
			std::unordered_set<size_t> _majorClassRulesCovered;
			for (auto _rule : minorClassRules) {
				for (auto _ruleCoveredMajorInstances : _rule.MajorInstancesCovered()) {
					_majorClassRulesCovered.insert(_ruleCoveredMajorInstances);
				}
			}
			for (size_t _majorInstanceCovered : _majorClassRulesCovered) {
				this->m_CoveredInstancesOfMajorClass.erase(_majorInstanceCovered);
			}
			// Special treatment if all instances are covered by rules of minor class.
			// In this situation we set the confidence and support to zero since the default
			// rule do not cover any instances of minor class.
			if (this->m_CoveredInstancesOfMajorClass.size() + this->m_CoveredInstancesOfMinorClass.size() == 0) {
				this->m_Confidence = 0;
				this->m_Support = 0;
			}
			else {
				this->_calculateRuleMetrics(true);
			}
		}

		// Construct a rule from an existing rule with a new term to be added.
		Rule(const Rule& rule, const Term& newTerm) {
			this->_copyFrom(rule);
			this->AddTerm(newTerm);
			this->_updateCoveredInstances();
			this->_calculateRuleMetrics();
		}
		// Copy constructor of Rule.
		Rule(const Rule& rule) { this->_copyFrom(rule); }

	public: // Attributes
		// Get the terms of this rule.
		const std::vector<Term>& Terms() const { return this->m_Terms; }
		// Get the indices of major class instances covered by the rule.
		const std::unordered_set<size_t>& MajorInstancesCovered() const { return this->m_CoveredInstancesOfMajorClass; }
		// Get the indices of minor class instances covered by the rule.
		const std::unordered_set<size_t>& MinorInstancesCovered() const { return this->m_CoveredInstancesOfMinorClass; }
		// Get the label of the rule
		const std::string& Label() const { return this->m_Label; }
		// Get the confidence of the rule.
		double Confidence() { return this->m_Confidence; }

	public: // Methods
		// Overload of operator=.
		Rule operator= (const Rule& rule) { return Rule(rule); }
		// Add a new term to this rule and update the covered instances.
		void AddTerm(const Term& newTerm) {
			this->m_Terms.push_back(Term(newTerm));
			this->_updateCoveredInstances();
			this->_calculateRuleMetrics();
		}
		// Returns whether this rule can cover a given instance.
		bool Cover(Dataset::Instance& instance) {
			for (size_t i = 0; i < this->m_Terms.size(); i++)
				if (this->m_Terms[i].Satisfy(instance) == false)
					return false;
			return true;
		}
		// Transform a rule to a std::string object. Equals to __str__() in python.
		std::string ToString(bool extraInfo = true) {
			std::string result;
			if (this->m_Terms.size() > 0) {
				// Not a default rule.
				std::vector<std::string> _termsStr;
				_termsStr.reserve(this->m_Terms.size());
				for (size_t i = 0; i < this->m_Terms.size(); i++) {
					_termsStr.push_back(m_Terms[i].ToString());
				}
				std::string delimiter = std::string(" and ");
				std::string _antecedents = Utils::join(delimiter, _termsStr);
				result.append(_antecedents).append(" => ").append(this->m_Label);
			}
			else {
				// This is a default rule.
				result.append(" => ").append(this->m_Label);
			}
			if (extraInfo)
				// Add coverage info.
				result.append(" (").append(
					Utils::integerToString((int)this->m_CoveredInstancesOfMinorClass.size())
				).append("/").append(
					Utils::integerToString((int)this->m_CoveredInstancesOfMajorClass.size())
				).append(")");
			return result;
		}
	};
public: // Constants
	static const int DecisionByAverageConfidence = 0;
	static const int DecisionByAnyMinorRule = 1;

	class ConfidenceCorrection {
		// Set of functions which
		// 1) symmetry at 0.5 (imbalance rate at 0.5 means the data are balanced)
		// 2) achieve maximum value 1 at 0.5 (this indicates that no correction will 
		//    be made when the data are balanced.)
		// 3) use a hyperparameter to control its value.
	public:
		static const int CorrectImbalanceByGaussianFunction = 0;
		static const int CorrectImbalanceByCosineH = 1;
		static double CorrectedConfidenceGaussian(double minimumConfidence, double imbRate, double sigma) {
			return minimumConfidence * exp(-1 * (imbRate - 0.5) * (imbRate - 0.5) / (2 * sigma * sigma));
		}

		static double CorrectedConfidenceCosineh(double minimumConfidence, double imbRate, double sigma) {
			return minimumConfidence * cos(imbRate - 0.5) * sigma;
		}
	};
	static const int CorrectImbalanceByGaussianFunction = 0;
	static const int CorrectImbalanceByCosineH = 1;
private: // Fields.
	// Hyper-Parameters of the rule-based classifier.
	double m_Shrinkage = 1.0;
	double m_MinimumConfidence = 0.90;
	size_t m_MinimumCoveragePerRule = 2;
	int m_DecisionMode;
	double m_DecisionThreshold;
	int m_CorrectionFunction = 0;
	double m_CorrectionFunctionParameter = 0.0;
	// Internal fields.
	std::list<Rule> m_Rules;
	double m_DefectiveRate;
	size_t m_MinorClass = -1;

public: // Constructors.
	// Constructor of ImbRuleInductionClassifier
	ImbRuleInductionClassifier(double shrinkage, double minimumConfidence, size_t minimumCoveragePerRule,
		int decisionMode = ImbRuleInductionClassifier::DecisionByAnyMinorRule,
		int confidenceCorrectionFunction = ImbRuleInductionClassifier::ConfidenceCorrection::CorrectImbalanceByGaussianFunction,
		double confidenceCorrectionParameter = 0.8) {
		this->m_Shrinkage = shrinkage;
		this->m_MinimumConfidence = minimumConfidence;
		this->m_MinimumCoveragePerRule = minimumCoveragePerRule;
		this->m_DefectiveRate = 0.0;
		this->m_DecisionMode = decisionMode;
		this->m_DecisionThreshold = 0.5;
		this->m_CorrectionFunction = confidenceCorrectionFunction;
		this->m_CorrectionFunctionParameter = confidenceCorrectionParameter;
	}
private:
	double _gainFunction(Rule& oldRule, Rule& newRule, const std::unordered_map<size_t, size_t>& instanceOccurrence) {
		size_t numMinorCoveredAfter = newRule.MinorInstancesCovered().size();
		size_t numMajorCoveredAfter = newRule.MajorInstancesCovered().size();
		size_t numMinorCoveredBefore = oldRule.MinorInstancesCovered().size();
		size_t numMajorCoveredBefore = oldRule.MajorInstancesCovered().size();
		// Calculate the gain.
		double t3 = log2(((double)numMinorCoveredAfter + 1) / ((double)numMajorCoveredAfter + (double)numMinorCoveredAfter + 1))
			- log2(((double)numMinorCoveredBefore + 1) / ((double)numMajorCoveredBefore + (double)numMinorCoveredBefore + 1));
		double t2 = 0.0;
		// Calculate the weights for rebalancing the data.
		// t2 = sum_{x_i \in X_{minor_new_covered}} exp{-1 \cdot shrinkage \cdot t_{x_i}}
		for (auto it : newRule.MinorInstancesCovered()) {
			t2 += exp(-1 * this->m_Shrinkage * (double)(instanceOccurrence.at(it)));
		}
		double t1 = 1 / ((double)numMajorCoveredAfter + 1);
		return t3 * t2 * t1;
	}

	double _gainWithNewTerm(Rule& baseRule, Term& newTerm, Dataset& dataset, size_t minorClass,
		std::unordered_map<size_t, size_t>& instanceOccurrences,
		size_t numMinorCoveredByBaseRule, size_t numMajorCoveredByBaseRule) {
		// Construct candidate rule with new term and calculate the gain with coverage of instances
		// of both major and minor class.
		// std::cout << std::ios::fixed << std::setprecision(4) << std::endl;
		// std::cout << newTerm.ToString() << std::endl;
		Rule tempRule(baseRule, newTerm);
		return this->_gainFunction(baseRule, tempRule, instanceOccurrences);
	}

	std::unordered_set<size_t> m_MinorInstancesCoveredByRules;
	std::unordered_set<size_t> m_MinorInstancesIgnoredAsNoise;
	std::unordered_map<size_t, size_t> m_InstanceOccurrences;
	std::set<double> m_NumericalValues;

	void _dropRule(Rule& rule) {
		for (auto _minorCoveredId : rule.MinorInstancesCovered()) {
			m_InstanceOccurrences[_minorCoveredId]++;
			m_MinorInstancesIgnoredAsNoise.insert(_minorCoveredId);
		}
	}

	// Accept a single rule.
	void _acceptRule(Rule& rule) {
		// First we need to check whether the rule introduce new instances of minor class to be covered.
		if (Utils::isSubset(rule.MinorInstancesCovered(), this->m_MinorInstancesCoveredByRules)) {
			// No new instances are introduced
			std::cout << "Info : No new instances will be introduced by " << rule.ToString() << std::endl;
			// In this situation, we check whether one of the existing rules' coverage is the subset of the candidate.
			bool _removeFlag = false;
			for (auto iter = this->m_Rules.begin(); iter != this->m_Rules.end(); ) {
				auto _rule = *iter;
				// Test whether the new rule 
				if (Utils::isSubset(_rule.MinorInstancesCovered(), rule.MinorInstancesCovered())) {
					// In this situation, the new rule can replace the current rule since the old (current) 
					// rule's coverage is the subset of new rule. Thus, we remove the current rule.
					std::cout << "Info: Removing unnecessary rule " << _rule.ToString() << std::endl;
					iter = this->m_Rules.erase(iter);
					_removeFlag = true;
				}
				else {
					// Move to next rule.
					iter++;
				}
			}
			if (_removeFlag == true) {
				// If such rule exist, replace them with the candidate rule.
				this->m_Rules.push_back(rule);
				std::cout << "Info: Use the candidate to replace the removed rules: " << std::endl;
				std::cout << "      " << rule.ToString() << std::endl;
			}
		}
		else {
			for (auto iter = this->m_Rules.begin(); iter != this->m_Rules.end(); ) {
				auto _rule = *iter;
				// Test whether the new rule 
				if (Utils::isSubset(_rule.MinorInstancesCovered(), rule.MinorInstancesCovered())) {
					// In this situation, the new rule can replace the current rule since the old (current) 
					// rule's coverage is the subset of new rule. Thus, we remove the current rule
					std::cout << "Info: Removing unnecessary rule " << _rule.ToString() << std::endl;
					iter = this->m_Rules.erase(iter);
				}
				else {
					// Move to next rule.
					iter++;
				}
			}
			// Add the new rule to ruleset.
			std::cout << "Info: Add rule: " << rule.ToString() << std::endl;
			this->m_Rules.push_back(rule);
		}
		// Update the occurrences and coverage status.
		for (auto _minorCoveredId : rule.MinorInstancesCovered()) {
			m_MinorInstancesCoveredByRules.insert(_minorCoveredId);
			m_InstanceOccurrences[_minorCoveredId]++;
		}
	}

	void _calcOptThreshold(Dataset& dataset) {
		std::vector<Dataset::Instance> instances;
		std::vector<std::string> _actualLabels;
		std::vector<double> _probas;
		std::vector<double> _thresholds;
		std::set<double> _uniqueProbas;
		// Get instances (train_x) and labels (train_y).
		for (size_t i = 0; i < dataset.InstanceCount(); i++) {
			auto _currentInstance = dataset.GetInstance(i);
			instances.push_back(_currentInstance);
			_actualLabels.push_back(_currentInstance.GetLabel());
		}
		// Get all possible thresholds.
		this->predictProba(instances, _probas);
		for (auto _proba : _probas) {
			_uniqueProbas.insert(_proba);
		}
		// Threshold = argmax_{thresholds} (F1Score).
		double _maxF1Score = 0.0;
		double _maxF1ScoreThreshold = 0.0;
		for (auto iter = _uniqueProbas.begin(); iter != _uniqueProbas.end(); ++iter) {
			std::vector<std::string> _predictedLabels;
			double _currentThreshold = *iter;
			this->_predictByAverageConfidence(instances, _currentThreshold, _predictedLabels);
			double _currentF1 = ConfusionMatrix(_predictedLabels, _actualLabels).F1Score();
			if (_currentF1 > _maxF1Score) {
				_maxF1Score = _currentF1;
				_maxF1ScoreThreshold = _currentThreshold;
			}
		}
		this->m_DecisionThreshold = _maxF1ScoreThreshold;
	}

	std::vector<std::string>& _predictBySingleRule(std::vector<Dataset::Instance> instances, std::vector<std::string>& results) {
		results.clear();
		results.reserve(instances.size());
		for (auto _instance : instances) {
			for (auto _rule : this->m_Rules) {
				if (_rule.Cover(_instance)) {
					results.push_back(_rule.Label());
					break;
				}
			}
		}
		return results;
	}

	std::vector<std::string>& _predictByAverageConfidence(std::vector<Dataset::Instance>& instances, double threshold, std::vector<std::string>& results) {
		std::vector<double> _probas;
		results.clear();
		results.reserve(instances.size());
		this->predictProba(instances, _probas);
		for (auto _proba : _probas) {
			results.push_back(_proba >= threshold ? "1" : "0");
		}
		return results;
	}

public: // Attributes
	// Returns the count of the rules.
	size_t RuleCount() { return this->m_Rules.size(); }
	// Returns the average length of the rules.
	double AverageRuleLength() {
		if (this->RuleCount() == 0) {
			Out::fatalError("Cannot calculate AverageRuleLength without training the classifier.");
		}
		double _sumLength = 0;
		for (auto rule : this->m_Rules) {
			_sumLength += rule.Terms().size();
		}
		return (double)_sumLength / (double)(this->RuleCount());
	}
	// Returns the maximum length of the rules.
	size_t MaximumRuleLength() {
		if (this->RuleCount() == 0) {
			Out::fatalError("Cannot calculate MaximumRuleLenngth without training the classifier.");
		}
		size_t _maxLength = 0;
		for (auto rule : this->m_Rules) {
			if (rule.Terms().size() > _maxLength) {
				_maxLength = rule.Terms().size();
			}
		}
		return _maxLength;
	}
	// Returns the rule set.
	const std::list<Rule>& Rules() const { return this->m_Rules; }
	// Returns the number of minor instances covered by the rules with correct labels.
	size_t CoveredMinorInstanceCount() { return this->m_MinorInstancesCoveredByRules.size(); }
	// Returns the number of minor instances covered by the rules with incorrect labels.
	size_t NoiseMinorInstanceCount() { return this->m_MinorInstancesIgnoredAsNoise.size(); }
public: // Methods
	void fit(Dataset& dataset, bool verbose = true) {
		// Init before the rule induction.
		this->m_Rules.clear();
		this->m_MinorInstancesCoveredByRules.clear();
		this->m_MinorInstancesIgnoredAsNoise.clear();
		this->m_InstanceOccurrences.clear();
		// First we need to find minor class
		size_t _instanceCount = dataset.InstanceCount();
		std::unordered_set<size_t> _buggyInstances;
		std::unordered_set<size_t> _cleanInstances;
		size_t _majorClass = 0;
		size_t _minorClass = 1;
		std::unordered_set<size_t>* _minorClassInstancesPtr = &_buggyInstances;
		std::unordered_set<size_t>* _majorClassInstancesPtr = &_cleanInstances;
		for (size_t i = 0; i < _instanceCount; i++) {
			Dataset::Instance _instance = dataset.GetInstance(i);
			if (ImbRuleInductionClassifier::LabelStringToUnsigned(_instance.GetLabel()) == 1) {
				_buggyInstances.insert(i);
			}
			else {
				_cleanInstances.insert(i);
			}
		}
		if (_buggyInstances.size() > _cleanInstances.size()) {
			_majorClass = 1;
			_minorClass = 0;
			_minorClassInstancesPtr = &_buggyInstances;
			_minorClassInstancesPtr = &_cleanInstances;
		}
		auto& _minorClassInstances = *(_minorClassInstancesPtr);
		auto& _majorClassInstances = *(_majorClassInstancesPtr);
		this->m_MinorClass = _minorClass;
		// Calculate the imbalance rate.
		double _imbalanceRate = (double)(_minorClassInstances.size())
			/ ((double)(_majorClassInstances.size()) + ((double)(_minorClassInstances.size())));
		this->m_DefectiveRate = (double)(_buggyInstances.size()) / (double)(_cleanInstances.size());
		// Initialize the occurrences for instances of minor class. We set all the instances of minor class
		// are never covered, i.e., (t[i] = 0 for all instance[i] is minor class).
		for (auto it = _minorClassInstances.begin(); it != _minorClassInstances.end(); ++it) {
			this->m_InstanceOccurrences.insert(std::pair<size_t, size_t>(*it, 0));
		}
		std::cout << "Info: Stats of training data. Num Buggy: " << _buggyInstances.size() << ", Num Clean: " << _cleanInstances.size() << std::endl;
		// Start rule induction from instances of minor class.
		while (m_MinorInstancesCoveredByRules.size() + m_MinorInstancesIgnoredAsNoise.size() < _minorClassInstances.size()) {
			
			// Output the weights of instances
			for (auto it = _minorClassInstances.begin(); it != _minorClassInstances.end(); ++it) {
				auto _instanceWeight = exp(-1 * this->m_Shrinkage * (double)(this->m_InstanceOccurrences.at(*it)));
				Dataset::Instance _instanceData = dataset.GetInstance(*it);
				std::cout << "Instance " << *it << ", label = " << _instanceData.GetLabel() << ", weight = " << _instanceWeight << std::endl;
			}

			if (verbose) {
				std::cout << "\nInfo: Ruleset stats. Rule count: " << this->m_Rules.size()
					<< ", covered minor:" << this->m_MinorInstancesCoveredByRules.size()
					<< ", noise: " << this->m_MinorInstancesIgnoredAsNoise.size()
					<< ", left: " << _minorClassInstances.size() - (m_MinorInstancesCoveredByRules.size() + m_MinorInstancesIgnoredAsNoise.size()) << std::endl;
			}

			// Stage 1: Find best split for numerical attributes and construct candidate term.
			// Prepare an empty rule for induction.
			Rule _currentRule(Utils::integerToString(_minorClass), nullptr, &dataset);
			// Build a single rule from an empty rule.
			while (true) {
				std::cout << "Info: The rule now covers (minor, major) instances " << std::endl;
				std::cout << "      " << _currentRule.MinorInstancesCovered().size() << ", " << _currentRule.MajorInstancesCovered().size() << std::endl;
				// We consider the initial maximum gain is 0.0 since the gain less than 0.0 will achieve worse result.
				double _maxGain = 0.0;
				std::string _maxTermAttribute;
				int _maxTermOperator = -1;
				double _maxTermValue = 0.0;
				// Enumerate all possible splits of all attributes and build the candidate term from the split with the maximum gain.
				for (size_t j = 0; j < dataset.Attributes().size(); j++) {
					std::string _selectedAttributeStr = dataset.Attributes()[j];
					// Get the non-duplicate values for attribute[j].
					m_NumericalValues.clear();
					for (auto _coveredMinorInstanceId : _currentRule.MinorInstancesCovered()) {
						m_NumericalValues.insert(dataset.GetValueAt(_coveredMinorInstanceId, _selectedAttributeStr));
					}
					// Calculate the gain of each split by comp
					bool _recordFirstVal = false;
					double _lastVal = DBL_MIN;
					for (auto _val : m_NumericalValues) {
						if (_recordFirstVal == false) {
							_recordFirstVal = true;
						}
						else {
							// Binary discretization.
							double _cutPoint = (_val + _lastVal) / 2;
							Term _leTerm(_selectedAttributeStr, Term::OperatorLessEqual, _cutPoint);
							double _leGain = this->_gainWithNewTerm(_currentRule, _leTerm, dataset, _minorClass, m_InstanceOccurrences,
								_currentRule.MinorInstancesCovered().size(), _currentRule.MajorInstancesCovered().size());
							if (_leGain > _maxGain) {
								_maxGain = _leGain;
								_maxTermAttribute = _leTerm.Attribute();
								_maxTermOperator = _leTerm.Operator();
								_maxTermValue = _leTerm.Value();
							}
							Term _geTerm(_selectedAttributeStr, Term::OperatorGreaterEqual, _cutPoint);
							double _geGain = this->_gainWithNewTerm(_currentRule, _geTerm, dataset, _minorClass, m_InstanceOccurrences,
								_currentRule.MinorInstancesCovered().size(), _currentRule.MajorInstancesCovered().size());
							if (_geGain > _maxGain) {
								_maxGain = _geGain;
								_maxTermAttribute = _geTerm.Attribute();
								_maxTermOperator = _geTerm.Operator();
								_maxTermValue = _geTerm.Value();
							}
						}
						_lastVal = _val;
					}
				}
				// Are the splits able to achieve better gain?
				// Assume 0.0001 is for comparing float values.
				if (_maxGain <= 0.0001) {
					// In this situation, no candidate term can achieve better gain.
					// Thus, the consturction of the rule ends.
					// We increase the occurrence of the minor class instances, and mark them as noise samples.
					this->_dropRule(_currentRule);
					if (verbose) {
						std::cout << "Info: Ignore Rule " << _currentRule.ToString() << " due to no best split." << std::endl;
					}
					break;
				}
				// Construct best term for building rule.
				Term _bestTerm(_maxTermAttribute, _maxTermOperator, _maxTermValue);
				std::cout << "Info: Maximum Gain is " << _maxGain << std::endl;
				std::cout << "      Corresponding term is " << _bestTerm.ToString() << std::endl;
				// We first add the "best" candidate term to the rule and update the coverage.
				_currentRule.AddTerm(_bestTerm);
				std::cout << "      After adding the term the candidate rule is" << std::endl;
				std::cout << "      " << _currentRule.ToString() << std::endl;
				double _currentConfidence = (double)(_currentRule.MinorInstancesCovered().size())
					/ ((double)(_currentRule.MinorInstancesCovered().size()) + (double)(_currentRule.MajorInstancesCovered().size()));
				// Test whether it covers instances more than the m_MinimumCoveragePerRule.
				if (_currentRule.MinorInstancesCovered().size() < this->m_MinimumCoveragePerRule) {
					// Too few instances of minor class are covered.
					this->_dropRule(_currentRule);
					std::cout << "Info: Ignore " << _currentRule.ToString() << " since too few instances of minor class are covered." << std::endl;
					// Go to next rule.
					break;
				}
				auto _correctFunctionPtr = this->m_CorrectionFunction == ImbRuleInductionClassifier::ConfidenceCorrection::CorrectImbalanceByGaussianFunction ?
					ImbRuleInductionClassifier::ConfidenceCorrection::CorrectedConfidenceGaussian : ImbRuleInductionClassifier::ConfidenceCorrection::CorrectedConfidenceCosineh;
				if (_currentConfidence >= (*_correctFunctionPtr)(this->m_MinimumConfidence, _imbalanceRate, this->m_CorrectionFunctionParameter)) {
					this->_acceptRule(_currentRule);
					// Go to next rule.
					break;
				}
				else {
					size_t _maxTermsToUse = dataset.Attributes().size() >= 4 ?
						(size_t)ceil(sqrt(dataset.Attributes().size())) : dataset.Attributes().size();
					//if (_currentRule.Terms().size() > _maxTermsToUse + 1) {
					if (_currentRule.Terms().size() > _maxTermsToUse + 1) {
						// More terms cannot be added. Mark them as noise.
						this->_dropRule(_currentRule);
						std::cout << "Info: Ignore Rule " << _currentRule.ToString() << " due to limitation of rule length." << std::endl;
						// Go to next rule.
						break;
					}
				}
			}
		}
		// Add a default rule to the final rule set.
		this->m_Rules.push_back(Rule(Utils::integerToString(_majorClass), this->m_Rules, &dataset));
		// Calculate the "optimal" threshold.
		this->_calcOptThreshold(dataset);
	}
	// Predict
	std::vector<std::string>& predict(std::vector<Dataset::Instance>& instances, std::vector<std::string>& results) {
		results.clear();
		results.reserve(instances.size());
		return this->m_DecisionMode == this->DecisionByAnyMinorRule ? this->_predictBySingleRule(instances, results) :
			this->_predictByAverageConfidence(instances, this->m_DecisionThreshold, results);
	}
	// Predict Proba
	std::vector<double>& predictProba(std::vector<Dataset::Instance> instances, std::vector<double>& results) {
		// Note that we only predict probability of label 1, i.e., being defective.
		results.clear();
		results.reserve(instances.size());
		for (auto _instance : instances) {
			double _sumConfidence = 0.0;
			size_t _numAppliedRules = 0;
			for (auto _rule : this->m_Rules) {
				if (_rule.Cover(_instance)) {
					// Rule type: default, non-default
					// Label of default rules is major class, while label of non-default rules is minor class.
					// Thus, 4 conditions need to be considered:
					// 1) Minor is 1, the rule is default rule => The confidence is the estimation of
					//    probability of being predicted as 0. => p(defective) = 1 - confidence.
					// 2) Minor is 1, the rule is non-default rule => The confidence is the estimation of
					//    probability of being predicted as 1. => p(defective) = confidence.
					// 3) Minor is 0, the rule is default rule => The confidence is the estimation of
					//    probability of being predicted as 1. => p(defective) = confidence.
					// 4) Minor is 0, the rule is non-default rule => The confidence is the estimation of
					//    probability of being predicted as 0. => p(defective) = 1 - confidence.
					if (this->m_MinorClass == 1) {
						_sumConfidence += _rule.isDefaultRule() ? (1 - _rule.Confidence()) : _rule.Confidence();
					}
					else { // if (this->m_MinorClass == 0) {
						_sumConfidence += _rule.isDefaultRule() ? _rule.Confidence() : (1 - _rule.Confidence());
					}
					_numAppliedRules++;
				}
			}
			results.push_back(_sumConfidence / _numAppliedRules);
		}
		return results;
	}
};

// Class for evaluate the rule induction algorithm.
class RuleEvaluator {
private: // Fields
	// Vector for pointers of ios objects.
	std::vector<std::ostream*> m_OutStreamPtrs;
	// Pointer of training set object.
	Dataset* m_TrainingSetPtr = nullptr;
	// Pointer of test set object;
	Dataset* m_TestSetPtr = nullptr;
	// Pointer of rule classifier object.
	ImbRuleInductionClassifier* m_RuleClassifierPtr = nullptr;

	std::vector<std::string> m_YGolden;
	std::vector<std::string> m_YPred;
	std::vector<double> m_YPredProba;
private: // Methods
	// Get the performance on the given dataset.
	ConfusionMatrix getPerformance(Dataset* datasetPtr) {
		m_YGolden.clear();
		m_YPred.clear();
		std::vector<Dataset::Instance> packedInstances;
		for (size_t i = 0; i < datasetPtr->InstanceCount(); i++) {
			packedInstances.push_back(datasetPtr->GetInstance(i));
			m_YGolden.push_back(datasetPtr->y(i));
		}
		this->m_RuleClassifierPtr->predict(packedInstances, m_YPred);
		this->m_RuleClassifierPtr->predictProba(packedInstances, m_YPredProba);
		return ConfusionMatrix(m_YPred, m_YGolden);
	}

	double getHyperParameters(std::unordered_map<std::string, double>& hyperParameters, std::string hyperParameterName, double defaultValue) {
		if (hyperParameters.find(hyperParameterName) != hyperParameters.end()) {
			return hyperParameters.at(hyperParameterName);
		}
		return defaultValue;
	}
public: // Methods
	// Constructor of RuleEvaluator.
	RuleEvaluator(std::ofstream* fileStreamPtr, Dataset* trainingSetPtr, Dataset* testSetPtr,
		std::unordered_map<std::string, double>& hyperParameters) {
		if (fileStreamPtr == nullptr) {
			Out::fatalError("Output filestream must be specified.");
		}
		if (trainingSetPtr == nullptr) {
			Out::fatalError("Training set must be specified.");
		}
		if (testSetPtr == nullptr) {
			Out::fatalError("Test set must be specified");
		}
		// Add output streams.
		m_OutStreamPtrs.push_back(&(std::cout));
		m_OutStreamPtrs.push_back(fileStreamPtr);
		// Set training data and test data.
		this->m_TrainingSetPtr = trainingSetPtr;
		this->m_TestSetPtr = testSetPtr;
		// Initialize the rule classifier.
		double _shrinkage = this->getHyperParameters(hyperParameters, std::string("--shrinkage"), 0.4);
		double _minimumConfidence = this->getHyperParameters(hyperParameters, std::string("--min-confidence"), 0.6);
		size_t _minimumInstancePerRule = (size_t)lround(this->getHyperParameters(hyperParameters, std::string("--min-covered"), 2));
		size_t _decisionMode = (size_t)lround(this->getHyperParameters(hyperParameters, std::string("--decision-mode"),
			ImbRuleInductionClassifier::DecisionByAverageConfidence));
		size_t _correctionFunction = (size_t)lround(this->getHyperParameters(hyperParameters, std::string("--correction"),
			ImbRuleInductionClassifier::ConfidenceCorrection::CorrectImbalanceByGaussianFunction));
		double _correctionParameter = this->getHyperParameters(hyperParameters, std::string("--correction-parameter"), 0.8);
		this->m_RuleClassifierPtr = new ImbRuleInductionClassifier(_shrinkage, _minimumConfidence, _minimumInstancePerRule,
			_decisionMode, _correctionFunction, _correctionParameter);
	}
	// Destructor.
	~RuleEvaluator() {
		delete this->m_RuleClassifierPtr;
	}
	// Building Classifier and output the rules.
	void BuildClassifier() {
		this->m_RuleClassifierPtr->fit(*(this->m_TrainingSetPtr));
		ConfusionMatrix cmat = this->getPerformance(this->m_TrainingSetPtr);
		for (std::ostream* _outStreamPtr : this->m_OutStreamPtrs) {
			std::ostream& _outStream = *(_outStreamPtr);
			_outStream << std::fixed << std::setprecision(3);
			_outStream << std::endl;
			_outStream << "=== Classifier model(full training set) ===" << std::endl;
			// Rules.
			_outStream << "Rules: " << std::endl;
			for (auto _rule : this->m_RuleClassifierPtr->Rules()) {
				_outStream << _rule.ToString() << std::endl;
				for (auto _instanceId : _rule.MinorInstancesCovered()) {
					_outStream << _instanceId << " ";
				}
				_outStream << std::endl;
			}
			// Statistics of Model Complexity.
			_outStream << std::endl << "Model Complexity Stats: " << std::endl;
			_outStream << "Num. Rules  " << this->m_RuleClassifierPtr->RuleCount() << std::endl;
			_outStream << "Avg. Len.   " << this->m_RuleClassifierPtr->AverageRuleLength() << std::endl;
			_outStream << "Max. Len.   " << this->m_RuleClassifierPtr->MaximumRuleLength() << std::endl;
			// Statisics of Coverage of Minor Instances.
			_outStream << "Minor Instances Stats:" << std::endl;
			size_t _numNoiseMinor = this->m_RuleClassifierPtr->NoiseMinorInstanceCount();
			size_t _numCoveredMinor = this->m_RuleClassifierPtr->CoveredMinorInstanceCount();
			_outStream << "# Total     " << (_numNoiseMinor + _numCoveredMinor) << std::endl;
			_outStream << "# Covered   " << _numCoveredMinor << std::endl;
			_outStream << "# Coverage  " << ((double)_numCoveredMinor / ((double)_numCoveredMinor + (double)_numNoiseMinor)) << std::endl;
			// Statistics of Performance on Training Set.
			_outStream << std::endl << "=== Evaluation on training set ===" << std::endl;
			_outStream << "Overall:" << std::endl;
			_outStream << "Accuracy    " << cmat.Accuracy() << std::endl;
			_outStream << "By Class:" << std::endl;
			_outStream << "Label           0        1\t\t" << std::endl;
			_outStream << "F1          " << cmat.F1Score(0) << "     " << cmat.F1Score(1) << std::endl;
			_outStream << "Recall      " << cmat.Recall(0) << "     " << cmat.Recall(1) << std::endl;
			_outStream << "Precision   " << cmat.Precision(0) << "     " << cmat.Precision(1) << std::endl;
			_outStream << "FP Rate     " << cmat.FalsePositiveRate(0) << "     " << cmat.FalsePositiveRate(1) << std::endl;
			_outStream << "ROC Area*   " << RuleAUC(cmat.Recall(0), cmat.FalsePositiveRate(0))
				<< "     " << RuleAUC(cmat.Recall(1), cmat.FalsePositiveRate(1)) << std::endl;

			_outStream << "Train Instances:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YGolden[i] << " ";
			_outStream << std::endl;
			_outStream << "Train Instances Predicted:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YPred[i] << " ";
			_outStream << std::endl;
			_outStream << "Train Instances Predicted Proba:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YPredProba[i] << " ";
			_outStream << std::endl;
		}
	}
	// Output on test set.
	void EvaluationOnTest() {
		ConfusionMatrix cmat = this->getPerformance(this->m_TestSetPtr);
		for (std::ostream* _outStreamPtr : this->m_OutStreamPtrs) {
			std::ostream& _outStream = *(_outStreamPtr);
			_outStream << std::endl << "=== Evaluation on test set ===" << std::endl;
			_outStream << "Overall:" << std::endl;
			_outStream << "Accuracy    " << cmat.Accuracy() << std::endl;
			_outStream << "By Class:" << std::endl;
			_outStream << "Label           0        1\t\t" << std::endl;
			_outStream << "F1          " << cmat.F1Score(0) << "     " << cmat.F1Score(1) << std::endl;
			_outStream << "Recall      " << cmat.Recall(0) << "     " << cmat.Recall(1) << std::endl;
			_outStream << "Precision   " << cmat.Precision(0) << "     " << cmat.Precision(1) << std::endl;
			_outStream << "FP Rate     " << cmat.FalsePositiveRate(0) << "     " << cmat.FalsePositiveRate(1) << std::endl;
			_outStream << "ROC Area*   " << RuleAUC(cmat.Recall(0), cmat.FalsePositiveRate(0))
				<< "     " << RuleAUC(cmat.Recall(1), cmat.FalsePositiveRate(1)) << std::endl;

			_outStream << "Test Instances:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YGolden[i] << " ";
			_outStream << std::endl;
			_outStream << "Test Instances Predicted:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YPred[i] << " ";
			_outStream << std::endl;
			_outStream << "Test Instances Predicted Proba:" << std::endl;
			for (size_t i = 0; i < m_YGolden.size(); i++)
				_outStream << m_YPredProba[i] << " ";
			_outStream << std::endl;
		}
	}
};


int main(int argc, const char* argv[]) {
	std::ios::sync_with_stdio(false);
	// Parse arguments.
	if (argc < 4) {
		Out::fatalError("Wrong argument(s). Usage: {train_file} {test_file} {output_file} [--shrinkage shrinkage] [--min-confidence min_confidence] [--min-coverage min_coverage]");
	}

	std::unordered_map<std::string, double> _hyperParameters;

	if (argc > 4) {
		if (argc % 2 > 0) {
			Out::fatalError("Wrong arguments(s). ");
		}
		for (size_t i = 4; i < argc && i + 1 < argc; i += 2) {
			std::string _hpName = argv[i];
			double _hpValue;
#ifdef _WIN32
			// MSVC will raise warning if use unsafe c functions.
			sscanf_s(argv[i + 1], "%lf", &_hpValue);
#else
			sscanf(argv[i + 1], "%f", &_hpValue);
#endif
			std::cout << "Set Hyperparameter " << _hpName << " = " << _hpValue << std::endl;
			_hyperParameters.insert(std::make_pair(_hpName, _hpValue));
		}
	}

	std::ofstream outputFile(argv[3], std::ofstream::trunc);
	if (outputFile.is_open() == false) {
		Out::fatalError("Failed to create output file");
	}

	std::cout << "Training set " << argv[1] << std::endl;
	std::cout << "Test set " << argv[2] << std::endl;

	Dataset dataset(argv[1]);
	Dataset testData(argv[2]);

	RuleEvaluator re(&outputFile, &dataset, &testData, _hyperParameters);
	re.BuildClassifier();
	re.EvaluationOnTest();
	outputFile.close();
}


