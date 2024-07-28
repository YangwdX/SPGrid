// --------------------------------------------------------------------
//
// title                  :gridding.cu
// description            :Gridding process.
// author                 :
//
// --------------------------------------------------------------------

#include "gridding_d.h"

double *d_lons;
double *d_lats;
double *d_data;
double *d_weights;

uint32_t *d_zyx;
double *d_xwcs;
double *d_ywcs;

// texture<float> tex_xwcs;
// texture<float> tex_ywcs;

double *d_datacube;
double *d_weightscube;

// uint32_t *h_searchcount;
uint32_t *h_computecount;
// uint32_t *d_searchcount;
uint32_t *d_computecount;

// uint32_t *h_toprow;
// uint32_t *h_bottomrow;
// uint32_t *d_toprow;
// uint32_t *d_bottomrow;
// uint32_t *h_mincol;
// uint32_t *h_maxcol;
// uint32_t *d_mincol;
// uint32_t *d_maxcol;

__constant__ uint32_t d_const_zyx[3];
__constant__ double d_const_kernel_params[3];
__constant__ GMaps d_const_GMaps;

// struct BufferLine{
//     int tag;                         //同时也是tag，缓存命中，禁止其他所有线程置换该行，但仍可读取和写入
//     int missLock;                    //未命中时，禁止其他所有线程读取该行，进行置换
//     double data[BLOCK_SIZE];
//     double weight[BLOCK_SIZE];
// };

/* Initializeoutput spectrals and weights. */
void init_output()
{
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    h_datacube = RALLOC(double, num);
    h_weightscube = RALLOC(double, num);
    for (uint32_t i = 0; i < num; ++i)
    {
        h_datacube[i] = 0.;
        h_weightscube[i] = 0.;
    }
}

/* Sinc function with simple singularity check. */
__device__ double sinc(double x)
{
    if (fabs(x) < 1.e-10)
        return 1.;
    else
        return sin(x) / x;
}

/* Grid-kernel definitions. get weight*/
__device__ double kernel_func_ptr(double distance, double bearing)
{
    if (d_const_GMaps.kernel_type == GAUSS1D)
    {                                                                // GAUSS1D
        return exp(-distance * distance * d_const_kernel_params[0]); // beam = 300, kernel_params[0]=199.62
    }
    else if (d_const_GMaps.kernel_type == GAUSS2D)
    { // GAUSS2D
        double ellarg = (pow(d_const_kernel_params[0], 2.0) * pow(sin(bearing - d_const_kernel_params[2]), 2.0) + pow(d_const_kernel_params[1], 2.0) * pow(cos(bearing - d_const_kernel_params[2]), 2.0));
        double Earg = pow(distance / d_const_kernel_params[0] /
                              d_const_kernel_params[1],
                          2.0) /
                      2. * ellarg;
        return exp(-Earg);
    }
    else if (d_const_GMaps.kernel_type == TAPERED_SINC)
    { // TAPERED_SINC
        double arg = PI * distance / d_const_kernel_params[0];
        return sinc(arg / d_const_kernel_params[2]) * exp(pow(-(arg / d_const_kernel_params[1]), 2.0));
    }
}

/*Binary search the start row and end row*/
__host__ __device__ uint32_t BiSearchRowBorder(double *ywcs, double *xwcs, uint32_t xcoord, uint32_t ycoord, double lats, double lons, uint32_t *top_row, uint32_t *bottom_row)
{

    int up = 0, down, mid;
    double distance;
    int locRow = ycoord - 1;
    double alpha = lons * DEG2RAD;
    double beta = lats * DEG2RAD;
    while (ywcs[locRow * xcoord] > lats && locRow > 0)
        locRow -= 1;
    down = locRow;
    while (up <= down)
    {
        mid = up + (down - up) / 2;
        // distance = true_angular_distance(alpha, beta, alpha, ywcs[mid * xcoord] * DEG2RAD) * RAD2DEG;
        distance = lats - ywcs[mid * xcoord];
        if (distance <= d_const_GMaps.sphere_radius)
        {
            down = mid - 1;
        }
        else
        {
            up = mid + 1;
        }
    }
    // if(down < 0) down = 0; // if up=down=0 initially, down will be -1
    //  down = max()
    *top_row = up;

    down = ycoord - 1;
    up = locRow + 1;
    while (up <= down)
    {
        mid = up + (down - up) / 2;
        // distance = true_angular_distance(alpha, beta, alpha, ywcs[mid * xcoord] * DEG2RAD) * RAD2DEG;
        distance = ywcs[mid * xcoord] - lats;
        if (distance <= d_const_GMaps.sphere_radius)
        {
            up = mid + 1;
        }
        else
        {
            down = mid - 1;
        }
    }
    // if(up > ycoord - 1) up = ycoord -1;
    *bottom_row = down;

    if (bottom_row >= top_row)
        return 1;
    else
        return 0;
}

/*Binary search the start col and end col*/
__host__ __device__ uint32_t BiSearchColBorder(double *ywcs, double *xwcs, uint32_t xcoord, uint32_t ycoord, uint32_t row, double lats, double lons, uint32_t *min_col, uint32_t *max_col)
{

    int left = 0, right, mid;
    double distance;
    int locCol = xcoord - 1;
    double alpha = lons * DEG2RAD;
    double beta = lats * DEG2RAD;
    while (xwcs[row * xcoord + locCol] < lons && locCol > 0)
        locCol -= 1;
    right = locCol;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        distance = true_angular_distance(alpha, beta, xwcs[row * xcoord + mid] * DEG2RAD, ywcs[row * xcoord + mid] * DEG2RAD) * RAD2DEG;
        if (distance <= d_const_GMaps.sphere_radius)
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    // if(right < 0) right = 0; // if left=right=0 initially, right will be -1
    *min_col = left;

    right = xcoord - 1;
    left = locCol + 1;
    while (left <= right)
    {
        mid = left + (right - left) / 2;
        distance = true_angular_distance(alpha, beta, xwcs[row * xcoord + mid] * DEG2RAD, ywcs[row * xcoord + mid] * DEG2RAD) * RAD2DEG;
        if (distance <= d_const_GMaps.sphere_radius)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    // if(up > ycoord - 1) up = ycoord -1;
    *max_col = right;
    ;

    if (max_col >= min_col)
        return 1;
    else
        return 0;
}
/*Search the start pix*/
__host__ __device__ uint32_t FindStartPoint(double *ywcs, double *xwcs, uint32_t xcoord, uint32_t ycoord, uint32_t ncoords, double ubound, double dbound, double lbound, double rbound)
{
    uint32_t sy = 0, sp;

    while (ywcs[sy] < ubound && sy < ncoords)
    {
        sy += xcoord;
    }
    if (sy > xcoord)
    {
        sy -= xcoord;
    }
    sp = sy;
    while (xwcs[sp] > lbound && sp < ncoords)
    {
        sp++;
    }
    if (sp > sy)
    {
        sp--;
    }
    return sp;
}

__global__ void hcgrid(
    double *d_lons,
    double *d_lats,
    double *d_data,
    double *d_weights,
    double *d_xwcs,
    double *d_ywcs,
    double *d_datacube,
    double *d_weightscube,
    uint32_t *d_computecount)
{

    uint32_t warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    // uint32_t tid = ((warp_id % d_const_GMaps.block_warp_num) * 32 + threadIdx.x % 32) * d_const_GMaps.factor;
    //  uint32_t warp_id = blockIdx.x * (blockDim.x * ((blockDim.y-1) / 32 + 1)) + threadIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    //  uint32_t tid = (warp_id - 1) * 32 + threadIdx.x % 32;
    //  uint32_t tid = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = gridDim.x * blockDim.x * blockDim.y;
    // printf("\nhere\n");
    // printf("%f\n", h_GMaps.sphere_radius);
    uint32_t idx, sp, op, top_row, bottom_row, min_col, max_col;
    uint32_t xcoord = d_const_zyx[2];
    uint32_t ycoord = d_const_zyx[1];
    uint32_t ncoords = d_const_zyx[1] * d_const_zyx[2];
    // uint32_t blocknum = ncoords / BLOCK_SIZE;

    idx = tid;

    for (idx = tid; idx < d_const_GMaps.data_shape; idx += total)
    {

        double alpha = d_lons[idx] * DEG2RAD;
        double beta = d_lats[idx] * DEG2RAD;
        double in_data = d_data[idx];
        double in_weights = d_weights[idx];

        double _radius = d_const_GMaps.sphere_radius * DEG2RAD; // While longitudes of the two points are the same, distance is equal to the difference of their latitudes
        double sphere_radius = d_const_GMaps.sphere_radius;
        // double lons_raidus = 2 * asin(sin(d_const_GMaps.sphere_radius / 2) / fabs(cos(beta)));
        // double ubound = d_lats[idx] - 1.1 * sphere_radius, dbound = d_lats[idx] + 1.1 * sphere_radius;
        // double lbound =  d_lons[idx] + 1.2 * sphere_radius, rbound = d_lons[idx] - 1.2* sphere_radius;
        // double ubound = (beta - _radius) * RAD2DEG, dbound = (beta + _radius) * RAD2DEG;
        //  double lbound = (alpha + (_radius/(cos(dbound * DEG2RAD)))) *RAD2DEG, rbound = (alpha -  (1.2*_radius)) * RAD2DEG;

        // int sp = FindStartPoint(d_ywcs, d_xwcs, xcoord, ycoord, ncoords, ubound, dbound, lbound, rbound);
        int status = BiSearchRowBorder(d_ywcs, d_xwcs, xcoord, ycoord, d_lats[idx], d_lons[idx], &top_row, &bottom_row);
        // d_toprow[idx] = top_row;
        // d_bottomrow[idx] = bottom_row;
        if (top_row > 0)
            top_row -= 1;
        if (bottom_row < ycoord - 1)
            bottom_row += 1;
        /* Gridding*/
        // if(status){
        for (int ri = top_row; ri <= bottom_row; ri++)
        {
            status = BiSearchColBorder(d_ywcs, d_xwcs, xcoord, ycoord, ri, d_lats[idx], d_lons[idx], &min_col, &max_col);
            // if(ri == top_row){
            //     d_mincol[idx] = min_col;
            //     d_maxcol[idx] = max_col;
            // } else {
            //     if(min_col < d_mincol[idx]) d_mincol[idx] = min_col;
            //     if(max_col > d_maxcol[idx]) d_maxcol[idx] = max_col;
            // }
            if (status)
            {
                if (min_col > 0)
                    min_col -= 1;
                if (max_col < xcoord - 1)
                    max_col += 1;
                for (int ci = min_col; ci <= max_col; ci++)
                {
                    int pixel = ri * xcoord + ci;
                    double ga = d_xwcs[pixel] * DEG2RAD;
                    double gb = d_ywcs[pixel] * DEG2RAD;
                    double sdist = true_angular_distance(alpha, beta, ga, gb) * RAD2DEG;
                    double sbear = 0.;
                    if (d_const_GMaps.bearing_needed)
                    {
                        sbear = great_circle_bearing(alpha, beta, ga, gb);
                    }
                    if (sdist < d_const_GMaps.sphere_radius)
                    {
                        double sweight = kernel_func_ptr(sdist, sbear);
                        double tweight = in_weights * sweight;
                        atomicAdd(&d_datacube[pixel], in_data * tweight);
                        atomicAdd(&d_weightscube[pixel], tweight);
                        // d_datacube[pixel] += in_data * tweight;
                        // d_weightscube[pixel] += tweight;
                        atomicAdd(&d_computecount[pixel], 1);
                        // d_computecount[pixel] += 1;
                    }
                }
            }
        }
        // }
    }
    return;
}

/* Alloc data for GPU. */
void data_alloc()
{
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    // uint32_t usedrings = h_Healpix.usedrings;
    HANDLE_ERROR(cudaMalloc((void **)&d_lons, sizeof(double) * data_shape));
    HANDLE_ERROR(cudaMalloc((void **)&d_lats, sizeof(double) * data_shape));
    HANDLE_ERROR(cudaMalloc((void **)&d_data, sizeof(double) * data_shape));
    HANDLE_ERROR(cudaMalloc((void **)&d_weights, sizeof(double) * data_shape));
    HANDLE_ERROR(cudaMalloc((void **)&d_xwcs, sizeof(double) * num));
    HANDLE_ERROR(cudaMalloc((void **)&d_ywcs, sizeof(double) * num));
    HANDLE_ERROR(cudaMalloc((void **)&d_datacube, sizeof(double) * num));
    HANDLE_ERROR(cudaMalloc((void **)&d_weightscube, sizeof(double) * num));

    // HANDLE_ERROR(cudaBindTexture(NULL, tex_xwcs, d_xwcs, sizeof(float)*num));
    // HANDLE_ERROR(cudaBindTexture(NULL, tex_ywcs, d_ywcs, sizeof(float)*num));

    // HANDLE_ERROR(cudaMalloc((void**)& d_hpx_idx, sizeof(uint64_t)*(data_shape+1)));
    //  HANDLE_ERROR(cudaMalloc((void**)& d_start_ring, sizeof(uint32_t)*(usedrings+1)));

    // HANDLE_ERROR(cudaMalloc((void**)& d_searchcount, sizeof(uint32_t)*data_shape));
    HANDLE_ERROR(cudaMalloc((void **)&d_computecount, sizeof(uint32_t) * num));
    // HANDLE_ERROR(cudaMalloc((void**)& d_toprow, sizeof(uint32_t)*data_shape));
    // HANDLE_ERROR(cudaMalloc((void**)& d_bottomrow, sizeof(uint32_t)*data_shape));
    // HANDLE_ERROR(cudaMalloc((void**)& d_mincol, sizeof(uint32_t)*data_shape));
    // HANDLE_ERROR(cudaMalloc((void**)& d_maxcol, sizeof(uint32_t)*data_shape));
}

/* Send data from CPU to GPU. */
void data_h2d()
{
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    // uint32_t usedrings = h_Healpix.usedrings;

    // Copy constants memory
    HANDLE_ERROR(cudaMemcpy(d_lons, h_lons, sizeof(double) * data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lats, h_lats, sizeof(double) * data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, sizeof(double) * data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weights, h_weights, sizeof(double) * data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_xwcs, h_xwcs, sizeof(double) * num, cudaMemcpyHostToDevice)); // float double
    HANDLE_ERROR(cudaMemcpy(d_ywcs, h_ywcs, sizeof(double) * num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_datacube, h_datacube, sizeof(double) * num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weightscube, h_weightscube, sizeof(double) * num, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpyAsync(d_hpx_idx, h_hpx_idx, sizeof(uint64_t)*(data_shape+1), cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(d_start_ring, h_start_ring, sizeof(uint32_t)*(usedrings+1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_kernel_params, h_kernel_params, sizeof(double) * 3));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_zyx, h_zyx, sizeof(uint32_t) * 3));
    // HANDLE_ERROR(cudaMemcpyToSymbol(d_const_Healpix, &h_Healpix, sizeof(Healpix)));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_GMaps, &h_GMaps, sizeof(GMaps)));
    // HANDLE_ERROR(cudaMemcpyAsync(d_toprow, h_toprow, sizeof(uint32_t)*data_shape, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpyAsync(d_bottomrow, h_bottomrow, sizeof(uint32_t)*data_shape, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpyAsync(d_mincol, h_mincol, sizeof(uint32_t)*data_shape, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpyAsync(d_maxcol, h_maxcol, sizeof(uint32_t)*data_shape, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpyAsync(d_searchcount, h_searchcount, sizeof(uint32_t)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyAsync(d_computecount, h_computecount, sizeof(uint32_t) * num, cudaMemcpyHostToDevice));
}

/* Send data from GPU to CPU. */
void data_d2h()
{
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];

    HANDLE_ERROR(cudaMemcpy(h_datacube, d_datacube, sizeof(double) * num, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_weightscube, d_weightscube, sizeof(double) * num, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(h_toprow, d_toprow, sizeof(uint32_t)*data_shape, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(h_bottomrow, d_bottomrow, sizeof(uint32_t)*data_shape, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(h_mincol, d_mincol, sizeof(uint32_t)*data_shape, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(h_maxcol, d_maxcol, sizeof(uint32_t)*data_shape, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(h_searchcount, d_searchcount, sizeof(uint32_t)*data_shape, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_computecount, d_computecount, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));
}
/* Release data. */
void data_free()
{
    DEALLOC(h_lons);
    HANDLE_ERROR(cudaFree(d_lons));
    DEALLOC(h_lats);
    HANDLE_ERROR(cudaFree(d_lats));
    DEALLOC(h_data);
    HANDLE_ERROR(cudaFree(d_data));
    DEALLOC(h_weights);
    HANDLE_ERROR(cudaFree(d_weights));
    DEALLOC(h_xwcs);
    HANDLE_ERROR(cudaFree(d_xwcs));
    DEALLOC(h_ywcs);
    HANDLE_ERROR(cudaFree(d_ywcs));
    // HANDLE_ERROR( cudaUnbindTexture(tex_xwcs) );
    // HANDLE_ERROR( cudaUnbindTexture(tex_ywcs) );
    DEALLOC(h_datacube);
    HANDLE_ERROR(cudaFree(d_datacube));
    DEALLOC(h_weightscube);
    HANDLE_ERROR(cudaFree(d_weightscube));
    // DEALLOC(h_hpx_idx);
    // HANDLE_ERROR( cudaFree(d_hpx_idx) );
    // DEALLOC(h_start_ring);
    // HANDLE_ERROR( cudaUnbindTexture(tex_start_ring) );

    // HANDLE_ERROR( cudaFree(d_start_ring) );
    DEALLOC(h_header);
    DEALLOC(h_zyx);
    DEALLOC(h_kernel_params);

    // DEALLOC(h_toprow);
    // DEALLOC(h_bottomrow);
    // DEALLOC(h_mincol);
    // DEALLOC(h_maxcol);
    // HANDLE_ERROR( cudaFree(d_toprow));
    // HANDLE_ERROR( cudaFree(d_bottomrow));
    // HANDLE_ERROR( cudaFree(d_mincol));
    // HANDLE_ERROR( cudaFree(d_maxcol));
    // HANDLE_ERROR( cudaFree(d_searchcount));
    HANDLE_ERROR(cudaFree(d_computecount));
    // DEALLOC(h_searchcount);
    DEALLOC(h_computecount);
}

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int &param, const int &bDim)
{
    double iTime1 = cpuSecond();
    // Read input points.
    // reah_input_map_hdf5(infile);
    // printf("\nhere\n");
    double iTime2 = cpuSecond();
    read_input_map(infile);
    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // hzyx[1]=ycoord  hzyx[2]=xcoord
    //  for(int i = 0; i < h_zyx[2]; i++){
    //      for(int j = i; j < h_zyx[1] * (h_zyx[2]-1); j += h_zyx[1])
    //          printf("%lf ", h_xwcs[j +  h_zyx[2]] -  h_xwcs[j]);     // ywcs 赤纬
    //      printf("\n\n");
    //  }
    //  for(int i = 0; i < h_zyx[1] * h_zyx[2]; i+= h_zyx[1]){
    //      for(int j = i; j < i + h_zyx[1] -1; j ++)
    //          printf("%lf ", h_ywcs[j] -  h_ywcs[j +1]);           // xwcs 赤经
    //      printf("\n");
    //  }
    //  printf("lons:\n");
    //  for(int i = 0; i < h_zyx[1]; i++){
    //      for(int j = 0; j < h_zyx[2]; j ++){
    //              printf("%lf ", h_xwcs[i * h_zyx[2] + j]);           // xwcs->lons 赤经
    //      }
    //      printf("\n\n");
    //  }

    // printf("lats:\n");
    // for(int i = 0; i < h_zyx[1]; i++){
    //     for(int j = 0; j < h_zyx[2]; j ++){
    //         printf("%lf ", h_ywcs[i * h_zyx[2] + j]);     // ywcs->lats 赤纬
    //     }
    //     printf("\n\n");
    // }
    // Initialize output spectrals and weights.
    init_output();

    double iTmie6 = cpuSecond();
    printf("pre process time = %f, ", (iTmie6 - iTime2) * 1000);

    double iTime3 = cpuSecond();
    // Alloc data for GPU.
    data_alloc();

    // h_searchcount = RALLOC(uint32_t, h_GMaps.data_shape);
    h_computecount = RALLOC(uint32_t, h_zyx[1] * h_zyx[2]);
    uint32_t pix_num = h_zyx[1] * h_zyx[2];
    // memset(h_searchcount, 0, sizeof(uint32_t)*h_GMaps.data_shape);
    memset(h_computecount, 0, sizeof(uint32_t) * h_zyx[1] * h_zyx[2]);
    // h_toprow = RALLOC(uint32_t, h_GMaps.data_shape);
    // h_bottomrow = RALLOC(uint32_t, h_GMaps.data_shape);
    // h_mincol = RALLOC(uint32_t, h_GMaps.data_shape);
    // h_maxcol = RALLOC(uint32_t, h_GMaps.data_shape);
    // memset(h_toprow, 0, sizeof(uint32_t)*h_GMaps.data_shape);
    // memset(h_bottomrow, 0, sizeof(uint32_t)*h_GMaps.data_shape);
    // memset(h_mincol, 0, sizeof(uint32_t)*h_GMaps.data_shape);
    // memset(h_maxcol, 0, sizeof(uint32_t)*h_GMaps.data_shape);

    double iTime4 = cpuSecond();
    printf("alloc time = %f\n", (iTime4 - iTime3) * 1000);

    // cudaDeviceProp prop;
    // checkRuntime(cudaGetDeviceProperties(&prop, 0));
    // printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);

    // Send data from CPU to GPU.
    data_h2d();
    printf("h_zyx[1]=%d, h_zyx[2]=%d, ", h_zyx[1], h_zyx[2]);

    double iTime7 = cpuSecond();
    printf("h2d time = %f\n", (iTime7 - iTime4) * 1000);

    // Set block and thread.
    // dim3 block(32,bDim);
    dim3 block(bDim);
    // dim3 grid((h_GMaps.block_warp_num * h_zyx[1] - 1) / (block.x / 32) + 1);
    dim3 grid((h_GMaps.data_shape - 1) / (block.x * block.y) + 1);
    // dim3 grid(336);
    printf("data size is %d\n", h_GMaps.data_shape);
    printf("grid.x=%d, block.x=%d, block.y=%d\n", grid.x, block.x, block.y);

    // Get start time.
    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Call device kernel.

    // HANDLE_ERROR(cudaFuncSetAttribute(hcgrid, cudaFuncAttributeMaxDynamicSharedMemorySize, CACHE_SIZE * sizeof(BufferLine)));
    // hcgrid<<<grid, block, CACHE_SIZE * sizeof(BufferLine)>>>(d_lons, d_lats, d_data, d_weights, d_xwcs, d_ywcs, d_datacube, d_weightscube);
    hcgrid<<<grid, block>>>(d_lons, d_lats, d_data, d_weights, d_xwcs, d_ywcs, d_datacube, d_weightscube, d_computecount);
    // Get stop time.
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("kernel elapsed time=%f, ", elapsedTime);

    double iTime8 = cpuSecond();
    // Send data from GPU to CPU
    data_d2h();
    double iTime9 = cpuSecond();
    printf("d2h time = %f\n", (iTime9 - iTime8) * 1000);

    // Write output FITS file
    write_output_map(outfile);

    // Write sorted input FITS file
    if (sortfile)
    {
        write_ordered_map(infile, sortfile);
    }

    // for(int i =  0; i <  h_GMaps.data_shape; i++){
    //     // printf("\nsample id= %d, top row = %d, bottom row = %d, min col = %d, max col = %d\n", i, h_toprow[i], h_bottomrow[i], h_mincol[i], h_maxcol[i]);
    //     if(h_maxcol[i] < 0 || h_maxcol[i] >= h_zyx[0] * h_zyx[1] * h_zyx[2] ){
    //         printf("\nsample id= %d,illegal id = %d, lats = %lf, lons = %lf\n", i, h_maxcol[i], h_lats[i], h_lons[i]);
    //     }
    //     // printf("\ncompute count min = %d, compute count max = %d, average ncompute count = %lf\n", cmin, cmax, avg_compute);
    // }
    uint32_t smin = h_GMaps.data_shape, cmin = h_GMaps.data_shape;
    uint32_t smax = 0, cmax = 0;
    double avg_search = 0, avg_compute = 0;

    for (int i = 0; i < pix_num; i++)
    {
        // if(h_searchcount[i] < smin) smin = h_searchcount[i];
        // if(h_searchcount[i] > smax) smax = h_searchcount[i];
        if (h_computecount[i] < cmin)
            cmin = h_computecount[i];
        if (h_computecount[i] > cmax)
            cmax = h_computecount[i];
        avg_compute += h_computecount[i];
        // avg_search += h_searchcount[i];
    }

    // avg_search = avg_search / h_GMaps.data_shape;
    avg_compute = avg_compute / (h_zyx[1] * h_zyx[2]);

    // double t_deg = 33.;
    // printf("%lf,%lf,%lf",t_deg,t_deg*DEG2RAD,t_deg*DEG2RAD*RAD2DEG);
    // printf("\nsearch count min = %d, search count max = %d, average search count = %lf\n", smin, smax, avg_search);
    printf("\ncompute count min = %d, compute count max = %d, average ncompute count = %lf\n", cmin, cmax, avg_compute);

    // Release data
    data_free();
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaDeviceReset());

    double iTime5 = cpuSecond();
    double iElaps = (iTime5 - iTime1) * 1000.;
    printf("solving_gridding time=%f\n", iElaps);
}