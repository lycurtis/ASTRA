#include <math.h>

/* This is a CUDA C kernel that calculates the cosine similarity between two float vectors.
   This kernel aims to compute cosine sim sequentially within a single GPU thread.
*/

__global__ void cosine_similarity_kernel(const float* vec_a, const float* vec_b, int dim, float* out_similarity) {
    float dot = 0.0f; //var for dot product
    float mag_a = 0.0f; //sum of squares for vect A's magnitude
    float mag_b = 0.0f; //sum of squares for vect B's magnitude

    /* This check ensures that even if this kernel is launched with many threads,
       only the very first thread (thread 0 in block 0) does the work.
       This makes it a simple, serial kernel suitable for being launched once per comparison.
    */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Loops through each element of the vectors 
        for (int i = 0; i < dim; ++i) {
            dot += vec_a[i] * vec_b[i]; // performs the dot product (a[i]*b[i]).
            mag_a += vec_a[i] * vec_a[i]; // sum of squares
            mag_b += vec_b[i] * vec_b[i];
        }
        mag_a = sqrtf(mag_a);
        mag_b = sqrtf(mag_b);

        // Only calculates the vectors if both vectors have a non zero magnitude
        if (mag_a > 0.0f && mag_b > 0.0f) {
            // We wanna write the result directly to output memory location on the GPU
            *out_similarity = dot / (mag_a * mag_b); // cosine sim formula
        } else {
            // If one or both vectors are zeros, similarity is then 0.
            *out_similarity = 0.0f;
        }
    }
}