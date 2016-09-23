#pragma once
#include <string>
#include <vector>

namespace utilities 
{
void Check(bool cond, const char* err);

enum StringSplitOptions {
    kStringSplitOptions_None,
    kStringSplitOptions_RemoveEmptyEntries
};

std::vector<std::string> Split(std::string target, char seperator, StringSplitOptions options = kStringSplitOptions_None);
std::vector<std::string> Split(std::string target, std::vector<char> seperator, StringSplitOptions options = kStringSplitOptions_None);
}