#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <cmath>
#include <boost/math/distributions/normal.hpp>

using namespace std;
using namespace arma;
using namespace mlpack::regression;


struct LogisticRegressionStats {
  vec coefficients;
  vec standardErrors;
  vec zStatistics;
  vec pValues;
};

mat CalculateHessian(const mat& X, const vec& params) {
  vec probabilities = 1 / (1 + exp(-X * params));
  vec w = probabilities % (1 - probabilities);
  mat W = diagmat(w);
  mat hessian = -X.t() * W * X;
  return 0.5 * (hessian + hessian.t()); // Ensure the matrix is symmetric
}


vec CalculateStandardErrors(const mat& hessian) {
  mat invHessian = inv(hessian);
  return sqrt(abs(invHessian.diag()));
}

vec CalculateZStatistics(const vec& coeffs, const vec& stdErrors) {
  return coeffs / stdErrors;
}

vec CalculatePValues(const vec& zStats) {
  vec pValues(zStats.n_elem);
  boost::math::normal_distribution<double> normalDist(0, 1);
  for (size_t i = 0; i < zStats.n_elem; ++i) {
    if (std::isnan(zStats(i))) {
      pValues(i) = arma::datum::nan;
    } else {
      pValues(i) = 2 * (1 - boost::math::cdf(normalDist, std::abs(zStats(i))));
    }
  }
  return pValues;
}
LogisticRegressionStats ComputeStatistics(const mat& inputData, const vec& coeffs) {
  // Add an intercept column to the inputData
  mat intercepts = ones<mat>(inputData.n_rows, 1);
  mat X = join_horiz(intercepts, inputData);

  mat hessian = CalculateHessian(X, coeffs);
  vec stdErrors = CalculateStandardErrors(hessian);
  vec zStats = CalculateZStatistics(coeffs, stdErrors);
  vec pValues = CalculatePValues(zStats);

  LogisticRegressionStats stats;
  stats.coefficients = coeffs;
  stats.standardErrors = stdErrors;
  stats.zStatistics = zStats;
  stats.pValues = pValues;

  return stats;
}



bool ReadCSV(const string& filename, mat& data) {
  try {
    data.load(filename, csv_ascii);
  } catch (const std::exception& e) {
    cout << "Error reading file: " << e.what() << endl;
    return false;
  }
  return true;
}
void PerformLogisticRegression(const mat& inputData, const mat& responsesData) {
  // Ensure the input data and responses have the same number of columns
  if (inputData.n_rows != responsesData.n_elem) {
    cout << "Error: input data and responses dimensions do not match." << endl;
    return;
  }

  // Convert responsesData to an arma::Row<size_t> object
  Row<size_t> responses = conv_to<Row<size_t>>::from(responsesData);

  // Train the logistic regression model
  LogisticRegression<> logisticRegression(inputData.t(), responses);


  LogisticRegressionStats stats = ComputeStatistics(inputData, logisticRegression.Parameters().t());

  cout << "Coefficients:\n" << stats.coefficients.t() << endl;
  cout << "Standard Errors:\n" << stats.standardErrors.t() << endl;
  cout << "Z-statistics:\n" << stats.zStatistics.t() << endl;
  cout << "P-values:\n" << stats.pValues.t() << endl;
}


int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage: " << argv[0] << " <data.csv> <responses.csv>" << endl;
    return 1;
  }

  // Read input data and responses
  mat inputData, responsesData;
  if (!ReadCSV(argv[1], inputData) || !ReadCSV(argv[2], responsesData)) {
    return 1;
  }
  // Perform logistic regression
  PerformLogisticRegression(inputData, responsesData);

  return 0;
}
