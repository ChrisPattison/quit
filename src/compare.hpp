#pragma once

bool FuzzyULPCompare(const float& a, const float& b, const int err = 10);

bool FuzzyULPCompare(const double& a, const double& b, const int err = 10);

bool FuzzyEpsCompare(const float& a, const float& b, const float err = 1e-6);

bool FuzzyEpsCompare(const double& a, const double& b, const double err = 1e-15);

bool FuzzyCompare(const double& a, const double& b);

bool FuzzyCompare(const float& a, const float& b);