/*
 * Copyright (C) 2016 Zhen Zhi Lee
 * Written by Zhen Zhi Lee (leezhenzhi@gmail.com)
 */

#include "BitArray.hpp"

BitArray::BitArray(const size_t& numberOfBits)
{
    // sizeof size_t = 4 bytes = 32 bits
    dataSize = numberOfBits / (8 * sizeof size_t) + 1;
    data = new size_t[dataSize];
    memset(data, 0, dataSize * sizeof size_t);
}

BitArray::~BitArray()
{
    delete[] data;
}

size_t* BitArray::GetData() const
{
    return data;
}

bool BitArray::operator[](const size_t& index) const
{
    auto i = index / (8 * sizeof size_t);
    auto pos = index % (8 * sizeof size_t);
    return data[i] & (1 << pos);
}

Bit BitArray::operator[](const size_t& index)
{
    auto i = index / (8 * sizeof size_t);
    auto pos = index % (8 * sizeof size_t);
    return Bit(&(data[i]), pos);
}
