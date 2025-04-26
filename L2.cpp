#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int> &arr)
{
    int n = arr.size();
    int temp;
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Optimized Parallel Bubble Sort
void bubbleSortParallel(vector<int> &arr)
{
    int n = arr.size();
    if (n < 2000)
    {
        bubbleSortSequential(arr);
        return;
    }
    int temp;
    for (int i = 0; i < n; i++)
    {
#pragma omp parallel for private(temp)
        for (int j = (i % 2 == 0) ? 0 : 1; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to generate random numbers
void generateRandomArray(vector<int> &arr)
{
    for (auto &x : arr)
    {
        x = rand() % 1000;
    }
}

int main()
{
    srand(time(0));
    omp_set_num_threads(4);

    vector<int> inputSizes(5);
    cout << "Enter 5 input sizes: ";
    for (int i = 0; i < 5; i++)
    {
        cin >> inputSizes[i];
    }

    cout << "+------------+---------------------+----------------------+-------------------+-------------------+\n";
    cout << "| Input Size           |  Seq Time         |  Par Time         |  Speedup       |  Efficiency |\n";
    cout << "+------------+---------------------+----------------------+-------------------+-------------------+\n";

    for (int t = 0; t < 5; t++)
    {
        int n = inputSizes[t];
        vector<int> arr1(n), arr2(n);

        generateRandomArray(arr1);
        arr2 = arr1; // Copy contents

        double start_time, end_time;

        start_time = omp_get_wtime();
        bubbleSortSequential(arr1);
        end_time = omp_get_wtime();
        double bubbleSortSeqTime = end_time - start_time;

        start_time = omp_get_wtime();
        bubbleSortParallel(arr2);
        end_time = omp_get_wtime();
        double bubbleSortParTime = end_time - start_time;

        double bubbleSortSpeedup = bubbleSortSeqTime / bubbleSortParTime;
        double bubbleSortEfficiency = bubbleSortSpeedup / 4;

        cout << "| "
             << setw(10) << n << " | "
             << setw(19) << fixed << setprecision(4) << bubbleSortSeqTime << " | "
             << setw(20) << fixed << setprecision(4) << bubbleSortParTime << " | "
             << setw(17) << fixed << setprecision(4) << bubbleSortSpeedup << " | "
             << setw(15) << fixed << setprecision(4) << bubbleSortEfficiency << " |\n";
    }

    cout << "+------------+---------------------+----------------------+-------------------+-------------------+\n";

    return 0;
}
