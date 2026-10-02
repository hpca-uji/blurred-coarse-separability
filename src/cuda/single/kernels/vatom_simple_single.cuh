#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "bresenham.cuh"
#include "pnoise.cuh"
#include <stdio.h>

extern "C"
{

    __device__ __forceinline__ float nextu(curandStatePhilox4_32_10_t *st, float ubuf[4], int *ui)
    {
        if (*ui == 4)
        {
            float4 r = curand_uniform4(st);
            ubuf[0] = r.x;
            ubuf[1] = r.y;
            ubuf[2] = r.z;
            ubuf[3] = r.w;
            *ui = 0;
        }
        return ubuf[(*ui)++];
    }

    __device__ void genvatom_simple_core(uint8_t *__restrict imgarr, double2 *__restrict vertex, uint8_t *__restrict colors,
                                         double2 *__restrict pos, int nvertex, int norbits,
                                         double ovalx, double ovaly, int freq1, int freq2,
                                         double noisecoef, int startrad, double linewidth, int res, int seed)

    {

        // Compute totalvertex
        int totalvertex = norbits * nvertex;

        // Generate position with only one thread
        if (threadIdx.x == 0)
        {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, totalvertex + norbits, 0ULL, &state);
            float4 u4 = curand_uniform4(&state);
            double x = (res + round((u4.x * 2 - 1) * res)) / 2;
            double y = (res + round((u4.y * 2 - 1) * res)) / 2;
            pos[0] = make_double2(x, y);
        }
        __syncthreads();

        // Generate color with multiple threads
        for (int orbitidx = threadIdx.x; orbitidx < norbits; orbitidx += blockDim.x)
        {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, totalvertex + orbitidx, 0ULL, &state);
            uint8_t gray = curand(&state) >> 24;
            colors[orbitidx] = gray;
        }
        __syncthreads();

        // Compute vertex
        const double angle = (2.0 * dpi) / nvertex;

        for (int vertexidx = threadIdx.x; vertexidx < nvertex; vertexidx += blockDim.x)
        {

            // Compute param angle
            const double paramangle = angle * vertexidx;
            double sinangle, cosangle;
            sincos(paramangle, &sinangle, &cosangle);

            // Compute base wave
            double wave = 0.5 * sin(paramangle * freq1) + 0.5 * sin(paramangle * freq2);

            // Compute initial vertex
            double vx = cosangle * startrad * ovalx + pos[0].x;
            double vy = sinangle * startrad * ovaly + pos[0].y;

            curandStatePhilox4_32_10_t st;
            curand_init(seed, vertexidx, 0ULL, &st);

            float ubuf[4];
            int ui = 4;

            // Compute per wave
            for (int orbitidx = 0; orbitidx < norbits; orbitidx++)
            {

                // Compute access id
                int id = nvertex * orbitidx + vertexidx;

                // Get uniforms
                float u1 = nextu(&st, ubuf, &ui);
                float u2 = nextu(&st, ubuf, &ui);

                // Compute noise
                double noise_x = noise1(u1 * 10000.0f, 1024, 0) * noisecoef - noisecoef - wave;
                double noise_y = noise1(u2 * 10000.0f, 1024, 0) * noisecoef - noisecoef - wave;

                // Apply modification
                vx -= (cosangle * (noise_x - linewidth));
                vy -= (sinangle * (noise_y - linewidth));

                // Store
                vertex[id].x = vx;
                vertex[id].y = vy;
            }
        }
        __syncthreads();

        // Draw sequentially per orbit
        for (int orbitidx = 0; orbitidx < norbits; orbitidx++)
        {
            for (int vertexidx = threadIdx.x; vertexidx < nvertex; vertexidx += blockDim.x)
            {
                // Compute access idx0
                int idx0 = orbitidx * nvertex + vertexidx;

                // Compute access idx1
                int next_vertex;
                if (vertexidx == nvertex - 1)
                {
                    next_vertex = 0;
                }
                else
                {
                    next_vertex = vertexidx + 1;
                }
                int idx1 = orbitidx * nvertex + next_vertex;

                int x0 = (int)vertex[idx0].x;
                int y0 = (int)vertex[idx0].y;
                int x1 = (int)vertex[idx1].x;
                int y1 = (int)vertex[idx1].y;

                draw_line(imgarr, x0, y0, x1, y1, res, colors[orbitidx]);
            }
            __syncthreads();
        }
    }

    __global__ void genvatom_simple_image(uint8_t *__restrict imgarr, double2 *__restrict vertex, uint8_t *__restrict colors,
                                          double2 *__restrict pos, int nvertex, int norbits,
                                          double ovalx, double ovaly, int freq1, int freq2,
                                          double noisecoef, int startrad, double linewidth, int res, int seed)
    {
        genvatom_simple_core(imgarr, vertex, colors, pos,
                             nvertex, norbits,
                             ovalx, ovaly,
                             freq1, freq2,
                             noisecoef, startrad,
                             linewidth, res, seed);
    }
}
