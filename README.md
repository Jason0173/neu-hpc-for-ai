# Assignment_1

The result is shown as below:



=thread-selected,id="1"
=== Running Correctness Tests ===

Test 1: 1x1 * 1x1
Expected: 6.00
Single-threaded: 6.00
Multi-threaded:  6.00
Results match: YES

Test 2: 1x1 * 1x5
Expected: [2.00 4.00 6.00 8.00 10.00]
Single-threaded: 2.00 4.00 6.00 8.00 10.00 
Multi-threaded:  2.00 4.00 6.00 8.00 10.00 
Results match: YES

Test 3: 2x1 * 1x3
Expected:
[3.00 4.00 5.00]
[6.00 8.00 10.00]
Single-threaded:
  3.00   4.00   5.00 
  6.00   8.00  10.00 

Multi-threaded:
  3.00   4.00   5.00 
  6.00   8.00  10.00 

Results match: YES

Test 4: 2x2 * 2x2
Expected:
[19.00 22.00]
[43.00 50.00]
Single-threaded:
 19.00  22.00 
 43.00  50.00 

Multi-threaded:
 19.00  22.00 
 43.00  50.00 

Results match: YES

Test 5: 3x4 * 4x3
Random matrices - checking consistency between single and multi-threaded
Results match: YES

Test 6: 1x3 * 3x1
Expected: 32.00
Single-threaded: 32.00
Multi-threaded:  32.00
Results match: YES

Test 7: 3x2 * 2x4
Expected:
[29.00 32.00 35.00 38.00]
[65.00 72.00 79.00 86.00]
[101.00 112.00 123.00 134.00]
Single-threaded:
 29.00  32.00  35.00  38.00 
 65.00  72.00  79.00  86.00 
101.00 112.00 123.00 134.00 

Multi-threaded:
 29.00  32.00  35.00  38.00 
 65.00  72.00  79.00  86.00 
101.00 112.00 123.00 134.00 

Results match: YES

=== All Correctness Tests Complete ===

=== Running Performance Tests ===

Matrix size: 100x100 * 100x100
Creating large random matrices...

Thread Count | Time (s) | Speedup | Efficiency
-------------|----------|---------|----------
           1 |    0.005 |    1.00 |     1.00
           4 |    0.002 |    3.07 |     0.77
          16 |    0.001 |    4.87 |     0.30
          32 |    0.002 |    2.86 |     0.09
          64 |    0.002 |    2.69 |     0.04
         128 |    0.002 |    2.88 |     0.02
Matrix size: 300x300 * 300x300
Creating large random matrices...

Thread Count | Time (s) | Speedup | Efficiency
-------------|----------|---------|----------
           1 |    0.081 |    1.00 |     1.00
           4 |    0.026 |    3.14 |     0.79
          16 |    0.019 |    4.30 |     0.27
          32 |    0.017 |    4.74 |     0.15
          64 |    0.017 |    4.80 |     0.07
         128 |    0.019 |    4.26 |     0.03

Matrix size: 500x500 * 500x500
Creating large random matrices...

Thread Count | Time (s) | Speedup | Efficiency
-------------|----------|---------|----------
           1 |    0.426 |    1.00 |     1.00
           4 |    0.123 |    3.47 |     0.87
          16 |    0.093 |    4.58 |     0.29
          32 |    0.088 |    4.83 |     0.15
          64 |    0.090 |    4.72 |     0.07
         128 |    0.109 |    3.90 |     0.03

Testing Complete         
