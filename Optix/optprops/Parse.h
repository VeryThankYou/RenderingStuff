/**
 * @file Parse.h
 * @brief Parsing various entites from a string.
 */

#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <sstream>
#include <vector>

std::string floatToString(float value);
void parse(const char* str,bool& x);
void parse(const char* str,std::string& x);
void parse(const char* str,int& x);
void parse(const char* str,float& x);
void parse(const char* str,std::vector<float>& v);
void parse(const char* str,std::vector<double>& v);
void parse(const char* str,std::vector<int>& v);

#endif // PARSE_H
