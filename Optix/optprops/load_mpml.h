#ifndef LOAD_MPML
#define LOAD_MPML

#include "Medium.h"
#include "Interface.h"

__declspec(dllexport)
void load_mpml(const std::string& filename);

__declspec(dllexport)
std::map<std::string, Medium>& get_media();

__declspec(dllexport)
std::map<std::string, Interface>& get_interfaces();

#endif // LOAD_MPML
