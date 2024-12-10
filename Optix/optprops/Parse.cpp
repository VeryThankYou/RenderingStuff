#include <cstring>
#include <cstdlib>
#include <iostream>
#include "Parse.h"

using namespace std;

namespace
{
	const char seps[]   = " ,\t\n\r";

	inline char* next_etoken(const char* btoken)
	{
		return const_cast<char*>(btoken)+strcspn(btoken, seps);
	}
	
	inline const char* next_btoken(char* etoken)
	{
		return const_cast<const char*>(etoken+strspn(etoken, seps));
	}
}

string floatToString(float value) {
	stringstream ret;
	string stringret;
	ret << value;
	getline(ret,stringret);
	return stringret;
}

void parse(const char* str,bool& x) {
	x = (strcmp(str,"true")==0) || (strcmp(str,"TRUE")==0);
}

void parse(const char* str,string& x) {
	x=string(str);
}

void parse(const char* str,int& x) {
	/* Establish string and get the first token: */
	x = strtol(str,0,10);
} 

void parse(const char* str,float& x) 
{
	/* Establish string and get the first token: */
	x = strtof(str,0);
}

void parse(const char* str,vector<float>& v) {
	const char *btoken=next_btoken(const_cast<char*>(str));
	char *etoken= next_etoken(btoken);
	while(etoken>btoken)
	{
		v.push_back(strtof(btoken,0));
		btoken=next_btoken(etoken);
		etoken=next_etoken(btoken);
	}
}

void parse(const char* str,vector<double>& v) {
	const char *btoken=next_btoken(const_cast<char*>(str));
	char *etoken= next_etoken(btoken);
	while(etoken>btoken)
	{
		v.push_back(strtod(btoken,0));
		btoken=next_btoken(etoken);
		etoken=next_etoken(btoken);
	}
}

void parse(const char* str,vector<int>& v) 
{
	const char *btoken=next_btoken(const_cast<char*>(str));
	char *etoken= next_etoken(btoken);
	while(etoken>btoken)
		{
			v.push_back(strtol(btoken,0,10));
			btoken=next_btoken(etoken);
			etoken=next_etoken(btoken);
		}
}
	
