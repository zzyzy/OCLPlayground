/*
 * Sieve of erostothenes example
 * 
 * Copyright (C) 2016 Zhen Zhi Lee
 * Written by Zhen Zhi Lee (leezhenzhi@gmail.com)
 * 
 * Demo of sieve of erostothenes
 * Adapted from:
 * http://www.geeksforgeeks.org/segmented-sieve/
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "BitArray.hpp"

std::vector<size_t> simpleSieve(size_t limit);
std::vector<size_t> segmentedSieve(size_t n);

int main()
{
    size_t endNumber = 100000;
    std::cout << "End number is " << endNumber << std::endl;

    auto primeNumbers = segmentedSieve(endNumber);
    std::ofstream outfile("PrimeNumbers.txt");
    for (auto p : primeNumbers)
    {
        std::cout << p << " ";
        outfile << p << " ";
    }
    std::cout << std::endl;
    outfile << std::endl;
    outfile.close();

    return 0;
}

std::vector<size_t> simpleSieve(size_t limit)
{
    std::vector<size_t> primeNumbers;
    BitArray sieve(limit + 1);

    primeNumbers.push_back(2);

    for (size_t p = 3; p * p <= limit; p += 2)
    {
        if (!sieve[p])
        {
            for (size_t i = p * p; i <= limit; i += p)
            {
                sieve[i] = true;
            }
        }
    }

    for (size_t i = 3; i <= limit; i += 2)
    {
        if (!sieve[i])
        {
            primeNumbers.push_back(i);
        }
    }

    return primeNumbers;
}

std::vector<size_t> segmentedSieve(size_t n)
{
    // Compute all primes smaller than or equal
    // to square root of n using simple sieve
    size_t limit = static_cast<size_t>(sqrt(n)) + 1;
    std::vector<size_t> initialPrimes = simpleSieve(limit);
    std::vector<size_t> primeNumbers = initialPrimes;
    size_t initialSize = initialPrimes.size();
    initialPrimes.clear();
    initialPrimes.shrink_to_fit();

    // Divide the range [0..n-1] in different segments
    // We have chosen segment size as sqrt(n).
    size_t low = limit;
    size_t high = 2 * limit;

    // While all segments of range [0..n-1] are not processed,
    // process one segment at a time
    while (low < n)
    {
        // To mark primes in current range. A value in mark[i]
        // will finally be false if 'i-low' is Not a prime,
        // else true.
        BitArray mark(limit + 1);

        // Use the found primes by simpleSieve() to find
        // primes in current range
        for (size_t i = 0; i < initialSize; ++i)
        {
            // Find the minimum number in [low..high] that is
            // a multiple of prime[i] (divisible by prime[i])
            // For example, if low is 31 and prime[i] is 3,
            // we start with 33.
            size_t loLim = static_cast<size_t>(low / primeNumbers[i]) * primeNumbers[i];
            if (loLim < low)
                loLim += primeNumbers[i];

            /*  Mark multiples of prime[i] in [low..high]:
            We are marking j - low for j, i.e. each number
            in range [low, high] is mapped to [0, high-low]
            so if range is [50, 100]  marking 50 corresponds
            to marking 0, marking 51 corresponds to 1 and
            so on. In this way we need to allocate space only
            for range  */
            for (size_t j = loLim; j < high; j += primeNumbers[i])
                mark[j - low] = true;
        }

        // Numbers which are not marked as false are prime
        for (size_t i = low; i < high; i++)
        {
            if (!mark[i - low])
            {
                //std::cout << i << "  ";
                primeNumbers.push_back(i);
            }
        }

        // Update low and high for next segment
        low = low + limit;
        high = high + limit;
        if (high >= n) high = n;
    }

    return primeNumbers;
}
