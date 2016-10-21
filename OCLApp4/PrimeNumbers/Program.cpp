#include <iostream>
#include <bitset>

#include "Bit.hpp"
#include "BitArray.hpp"

int main()
{
    int startNumber, endNumber;

    std::cout << "Enter the start number: ";
    std::cin >> startNumber;
    while (startNumber < 2)
    {
        std::cout << "Start number cannot be less than 2" << std::endl;
        std::cout << "Enter the start number: ";
        std::cin >> startNumber;
    }

    std::cout << "Enter the end number: ";
    std::cin >> endNumber;
    while (endNumber <= startNumber)
    {
        std::cout << "End number cannot be less than start number" << std::endl;
        std::cout << "Enter the end number: ";
        std::cin >> endNumber;
    }

    //bool* sieve = new bool[endNumber + 1];
    BitArray sieve(endNumber);
    //std::bitset<500001> sieve;

    for (auto i = 0; i < endNumber + 1; ++i)
    {
        sieve[i] = false;
    }

    for (auto p = 2; p * p <= endNumber; ++p)
    {
        if (!sieve[p])
        {
            for (auto i = p * 2; i <= endNumber; i += p)
            {
                sieve[i] = true;
            }
        }
    }

    std::cout << "Prime numbers: ";
    for (auto i = 0; i < endNumber + 1; ++i)
    {
        if (!sieve[i])
        {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    std::cin.get();

    //delete[] sieve;

    return 0;
}
