#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>
#include <list>

std::string trim(const std::string& s, const std::string& wspaces);
std::string trim(const std::string& s);
void split(const std::string& s, std::list<std::string>& result, const std::string& delim);
void split(const std::string& s, std::list<std::string>& result);
void trim_split(const std::string& s, std::list<std::string>& result, const std::string& delim);
void trim_split(const std::string& s, std::list<std::string>& result);
void get_first(std::string& s, std::string& first);
void get_last(std::string& s, std::string& last);

#endif // STRING_UTILS_H