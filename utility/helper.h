/*
 * helper.h
 *
 *  Created on: Jul 11, 2011
 *      Author: davheld
 */

#ifndef HELPER_H_
#define HELPER_H_

#include <string>
#include <iostream>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Convenience helper functions.

// *******Number / string conversions*************

// Conversions from number into a string.
std::string num2str(const int num);
std::string num2str(const float num);
std::string num2str(const double num);
std::string num2str(const double num, const int decimal_places);
std::string num2str(const unsigned int num);
std::string num2str(const size_t num);

// Conversions from string into a number.
template<class T>
  T str2num(const std::string& s);


// Template implementation
template<class T>
    T str2num(const std::string& s)
{
     std::istringstream stream (s);
     T t;
     stream >> t;
     return t;
}

#endif /* HELPER_H_ */

