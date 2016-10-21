/*
 * An array of bits
 * 
 * Copyright (C) 2016 Zhen Zhi Lee
 * Written by Zhen Zhi Lee (leezhenzhi@gmail.com)
 * 
 * Represents an array of bits, and provides
 * methods to manipulate the individual bits.
 * Unlike std::bitset, the size of the array
 * can be defined at runtime instead of compile
 * time. Unlike std::vector<bool>, this guarantees
 * that it is space efficient and it doesn't allow
 * dynamic resizing at runtime.
 * 
 * Adapted from:
 * http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html
 */

#pragma once
#ifndef __BIT_ARRAY_HPP__
#define __BIT_ARRAY_HPP__

#include "Bit.hpp"

class BitArray
{
public:
    explicit BitArray(const size_t& numberOfBits);
    ~BitArray();

    size_t* GetData() const;

    bool operator[](const size_t& index) const;
    Bit operator[](const size_t& index);

private:
    size_t* data;
};

#endif // __BIT_ARRAY_HPP__
