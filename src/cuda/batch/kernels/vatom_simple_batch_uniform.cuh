#include "vatom_simple_single_uniform.cuh"
#include <iostream>

extern "C"
{

    __global__ void genvatom_simple_batch(uint8_t *d_imgarr_base, double2 *d_vertex_base,
                                          uint8_t *d_colors_base, double2 *d_pos_base,
                                          int maxtotalvertex, int norbits_max,
                                          int *nvertex, int *norbits, double *ovalx,
                                          double *ovaly, int *freq1, int *freq2, double *noisecoef,
                                          int *startrad, double *linewidth, int res,
                                          int base_seed, int batch_size, int *classes_idx)
    {

        // The index of the image to generate
        const int img_idx = blockIdx.x;

        // Calculate the offsets for the current image's data buffers
        // Sizes for one image
        const size_t img_offset = (size_t)img_idx * res * res;
        const size_t vertex_offset = (size_t)img_idx * maxtotalvertex;
        const size_t colors_offset = (size_t)img_idx * norbits_max;
        const size_t pos_offset = (size_t)img_idx;

        // Pointers for the current image's data
        uint8_t *imgarr_ptr = d_imgarr_base + img_offset;
        double2 *vertex_ptr = d_vertex_base + vertex_offset;
        uint8_t *colors_ptr = d_colors_base + colors_offset;
        double2 *pos_ptr = d_pos_base + pos_offset;

        // Use a unique seed for each image
        int current_seed = base_seed * batch_size + img_idx;
        int class_idx = classes_idx[img_idx];
        genvatom_simple_core(imgarr_ptr, vertex_ptr, colors_ptr,
                             pos_ptr, nvertex[class_idx], norbits[class_idx],
                             ovalx[class_idx], ovaly[class_idx], freq1[class_idx],
                             freq2[class_idx], noisecoef[class_idx], startrad[class_idx],
                             linewidth[class_idx], res, current_seed);
    }
}