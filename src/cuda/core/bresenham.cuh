#include <cuda_fp16.h>
#include <fstream>
#pragma once
extern "C"
{

    __device__ void draw_line(uint8_t *__restrict imgarr, int x0, int y0, int x1, int y1, int res, uint8_t gray)
    {

        int i, n, e;
        int dx, dy;
        int xs, ys;

        /* normalize coordinates */
        dx = x1 - x0;
        if (dx < 0)
        {
            dx = -dx, xs = -1;
        }
        else
        {
            xs = 1;
        }
        dy = y1 - y0;
        if (dy < 0)
        {
            dy = -dy, ys = -1;
        }
        else
        {
            ys = 1;
        }

        n = (dx > dy) ? dx : dy;

        if (dx == 0)
        {
            /* vertical */
            for (i = 0; i < dy; i++)
            {
                if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                {
                    imgarr[y0 * res + x0] = gray;
                }
                y0 += ys;
            }
        }
        else if (dy == 0)
        {
            /* horizontal */
            for (i = 0; i < dx; i++)
            {
                if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                {
                    imgarr[y0 * res + x0] = gray;
                }
                x0 += xs;
            }
        }
        else if (dx > dy)
        {
            /* bresenham, horizontal slope */
            n = dx;
            dy += dy;
            e = dy - dx;
            dx += dx;

            for (i = 0; i < n; i++)
            {
                if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                {
                    imgarr[y0 * res + x0] = gray;
                }
                if (e >= 0)
                {
                    y0 += ys;
                    e -= dx;
                }
                e += dy;
                x0 += xs;
            }
        }
        else
        {
            /* bresenham, vertical slope */
            n = dy;
            dx += dx;
            e = dx - dy;
            dy += dy;

            for (i = 0; i < n; i++)
            {
                if (x0 >= 0 && x0 < res && y0 >= 0 && y0 < res)
                {
                    imgarr[y0 * res + x0] = gray;
                }
                if (e >= 0)
                {
                    x0 += xs;
                    e -= dy;
                }
                e += dx;
                y0 += ys;
            }
        }
    }
}