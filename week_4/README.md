chapter_6
q4:
Let's assume that each float is 4 bytes.

a.
OP/B = 2N/(2N+1)*4 bytes = 1/4

b.
tile = 32*32   tiling + shared memory
FLOPS = 2NT^2
bytes = (N/T) * 2*T^2*4 = 8NT
OP/B = (2NT^2)/(8NT)
as T = 32
OP/B = 8N/(N+16) when N is large the answer is 8;

c.
coarsening factor = 4
OP/B = (2NT^2)/(8NT)
as T = 32
OP/B = 8N/(N+16) when N is large the answer is 8;