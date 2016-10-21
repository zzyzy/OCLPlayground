/*
 * Represents a single bit
 *
 * Copyright (C) 2016 Zhen Zhi Lee
 * Written by Zhen Zhi Lee (leezhenzhi@gmail.com)
 *
 * Used together with BitArray. It references
 * a size_t element from the BitArray.
 *
 * Adapted from:
 * http://www.mathcs.emory.edu/~cheung/Courses/255/Syllabus/1-C-intro/bit-array.html
 */

#pragma once
#ifndef __BIT_HPP__
#define __BIT_HPP__

#include <iostream>

class Bit
{
public:
    Bit(size_t* data, const size_t& pos);
    ~Bit();

    size_t GetData() const;

    void operator=(const bool& bit) const;
    friend std::ostream& operator<<(std::ostream& out, const Bit& bit);
    operator bool() const;

private:
    size_t* data;
    size_t pos;
};

#endif // __BIT_HPP__
