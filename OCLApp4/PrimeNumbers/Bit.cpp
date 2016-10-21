/*
* Copyright (C) 2016 Zhen Zhi Lee
* Written by Zhen Zhi Lee (leezhenzhi@gmail.com)
*/

#include "Bit.hpp"
#include <bitset>

Bit::Bit(size_t* data, const size_t& pos) :
    data(data),
    pos(pos)
{
}

Bit::~Bit()
{
}

size_t Bit::GetData() const
{
    return *data;
}

void Bit::operator=(const bool& bit) const
{
    if (bit)
        *data |= 1 << pos;
    else
        *data &= ~(1 << pos);
}

std::ostream& operator<<(std::ostream& out, const Bit& bit)
{
    out << std::bitset<sizeof size_t * 8>(*bit.data);
    return out;
}

Bit::operator bool() const
{
    return (*data & (1 << pos));
}
