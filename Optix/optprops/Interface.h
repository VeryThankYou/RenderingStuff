#ifndef INTERFACE_H
#define INTERFACE_H

#include <map>
#include <string>
#include "Medium.h"

class Interface
{
public:
  Interface() : med_in(0), med_out(0) { }

  std::string name;
  Medium* med_in;
  Medium* med_out;
};

extern std::map<std::string, Interface> interfaces;

#endif
