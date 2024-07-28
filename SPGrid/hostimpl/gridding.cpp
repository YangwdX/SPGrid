// --------------------------------------------------------------------
//
// title                  :gridding.cu
// description            :Gridding process.
// author                 :
//
// --------------------------------------------------------------------

#include "gridding.h"
mutex globalMutex;

uint32_t *searchcount;
uint32_t *computecount;
/* Initialize output spectrals and weights. */
void init_output(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    h_datacube = RALLOC(double, num);
    h_weightscube = RALLOC(double, num);
    for(uint32_t i = 0; i < num; ++i){
        h_datacube[i] = 0.;
        h_weightscube[i] = 0.;
    }
}

/* Sinc function with simple singularity check. */
double sinc(double x){
    if(fabs(x) < 1.e-10)
        return 1.;
    else
        return sin(x) / x;
}

/* Grid-kernel definitions. get weight*/
double kernel_func_ptr(double distance, double bearing){
    if(h_GMaps.kernel_type == GAUSS1D){   // GAUSS1D
        return exp(-distance * distance * h_kernel_params[0]);
    }
    else if(h_GMaps.kernel_type == GAUSS2D){  // GAUSS2D
        double ellarg = (\
                pow(h_kernel_params[0], 2.0)\
                    * pow(sin(bearing - h_kernel_params[2]), 2.0)\
                + pow(h_kernel_params[1], 2.0)\
                    * pow(cos(bearing - h_kernel_params[2]), 2.0));
        double Earg = pow(distance / h_kernel_params[0] /\
                       h_kernel_params[1], 2.0) / 2. * ellarg;
        return exp(-Earg);
    }
    else if(h_GMaps.kernel_type == TAPERED_SINC){ // TAPERED_SINC
        double arg = PI * distance / h_kernel_params[0];
        return sinc(arg / h_kernel_params[2])\
            * exp(pow(-(arg / h_kernel_params[1]), 2.0));
    }
    return 0.0;
}
/*Binary search the start row and end row*/
uint32_t BiSearchRowBorder(double *ywcs,double *xwcs, uint32_t xcoord, uint32_t ycoord, double lats, double lons, uint32_t *top_row, uint32_t *bottom_row){
    
    int up = 0, down, mid; 
    double distance;  
    int locRow = ycoord -1;
    double alpha = lons * DEG2RAD;
    double beta = lats * DEG2RAD;
    while(ywcs[locRow * xcoord] > lats && locRow > 0) locRow -= 1;
    down = locRow;
    while(up <= down){
        mid = up + (down-up) / 2;
        // distance = true_angular_distance(alpha, beta, alpha, ywcs[mid * xcoord] * DEG2RAD) * RAD2DEG;
        distance = lats - ywcs[mid * xcoord];
        if(distance <= h_GMaps.sphere_radius){
            down = mid - 1;
        } else {
            up = mid + 1;
        }
    } 
    //if(down < 0) down = 0; // if up=down=0 initially, down will be -1
    // down = max()
    *top_row = up;

    down = ycoord - 1;
    up = locRow + 1;
    while(up <= down){
        mid = up + (down-up) / 2;
        // distance = true_angular_distance(alpha, beta, alpha, ywcs[mid * xcoord] * DEG2RAD) * RAD2DEG;
        distance = ywcs[mid * xcoord] - lats;
        if(distance <= h_GMaps.sphere_radius){
            up = mid + 1;
        } else {
            down = mid - 1;  
        }
    } 
    // if(up > ycoord - 1) up = ycoord -1;
    *bottom_row = down;
    
    if(bottom_row >= top_row) 
        return 1;
    else
        return 0;
}

/*Binary search the start col and end col*/
uint32_t BiSearchColBorder(double *ywcs,double *xwcs, uint32_t xcoord, uint32_t ycoord, uint32_t row, double lats, double lons, uint32_t *min_col, uint32_t *max_col){
    
    int left = 0, right, mid; 
    double distance;  
    int locCol = xcoord -1;
    double alpha = lons * DEG2RAD;
    double beta = lats * DEG2RAD;
    while(xwcs[row * xcoord + locCol] < lons && locCol > 0) locCol -= 1;
    right = locCol;
    while(left <= right){
        mid = left + (right - left) / 2;
        distance = true_angular_distance(alpha, beta, xwcs[row * xcoord + mid] * DEG2RAD, ywcs[row * xcoord + mid] * DEG2RAD) * RAD2DEG;
        if(distance <= h_GMaps.sphere_radius){
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    } 
    // if(right < 0) right = 0; // if left=right=0 initially, right will be -1
    *min_col = left;

    right = xcoord - 1;
    left = locCol + 1;
    while(left <= right){
        mid = left + (right - left) / 2;
        distance = true_angular_distance(alpha, beta, xwcs[row * xcoord + mid] * DEG2RAD, ywcs[row * xcoord + mid] * DEG2RAD) * RAD2DEG;
        if(distance <= h_GMaps.sphere_radius){
             left = mid + 1;
        } else {
            right = mid - 1;  
        }
    } 
    // if(up > ycoord - 1) up = ycoord -1;
    *max_col = right;;
    
    if(max_col >= min_col) 
        return 1;
    else
        return 0;
}

void hcgridThreaded (
        double *h_lons,
        double *h_lats,
        double *h_data,
        double *h_weights,
        double *h_xwcs,
        double *h_ywcs,
        double *localDataCube,
        double *localWeightsCube,
        uint32_t startIdx,
        uint32_t endIdx) {
        // uint32_t warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        // uint32_t tid = ((warp_id % h_GMaps.block_warp_num) * 32 + threadIdx.x % 32) * h_GMaps.factor;
        // printf("\nhere\n"); 
        // printf("%f\n", h_GMaps.sphere_radius);
        uint32_t idx;
        uint32_t xcoord = h_zyx[2];
        uint32_t ycoord = h_zyx[1];
        uint32_t ncoords = h_zyx[1] * h_zyx[2];
        // printf("localDataCube.size()=%ld, ", localDataCube.size());
        // printf("localWeightsCube.size()=%ld\n", localWeightsCube.size());
        for(uint32_t i = 0; i < ncoords; ++i){
            localDataCube[i] = 0.;
            localWeightsCube[i] = 0.;
        }
        for(idx = startIdx; idx < endIdx; idx ++){
            // printf("%d\n", idx);
            // int thread_id = omp_get_thread_num();
            double alpha = h_lons[idx] * DEG2RAD;
            double beta = h_lats[idx] * DEG2RAD;
            double in_data = h_data[idx];
            double in_weights = h_weights[idx];
            
            uint32_t top_row, bottom_row, max_col, min_col;
           
            int status = BiSearchRowBorder(h_ywcs, h_xwcs, xcoord, ycoord, h_lats[idx], h_lons[idx], &top_row, &bottom_row);
            if(top_row > 0) top_row -= 1;
            if(bottom_row < ycoord - 1) bottom_row += 1;
            // printf("top row is %d, bottom_row is %d, status is %d\n", top_row, bottom_row, status);

            /* Gridding*/
            for(int ri = top_row; ri <= bottom_row; ri ++){
                status = BiSearchColBorder(h_ywcs, h_xwcs, xcoord, ycoord, ri, h_lats[idx], h_lons[idx], &min_col, &max_col);
                
                if(min_col > 0) min_col -= 1;
                if(max_col < xcoord - 1) max_col += 1;
                for(int ci = min_col; ci <= max_col; ci ++){
                    int pixel = ri * xcoord + ci;
                    double ga = h_xwcs[pixel] * DEG2RAD;
                    double  gb = h_ywcs[pixel] * DEG2RAD;
                    double sdist = true_angular_distance(alpha, beta, ga, gb) * RAD2DEG;
                    // printf("%lf\n",sdist);
                    double sbear = 0.;
                    if (h_GMaps.bearing_needed) {
                        sbear = great_circle_bearing(alpha, beta, ga, gb);
                    }
                    if(sdist < h_GMaps.sphere_radius){
                        double sweight = kernel_func_ptr(sdist, sbear);
                        double tweight = in_weights * sweight;

                        // computecount[pixel]++;

                        // #pragma omp  atomic
                    
                        localDataCube[pixel] += in_data * tweight;
                        localWeightsCube[pixel] += tweight;
                    
                        // #pragma omp  atomic
                        // localDataCube[pixel] += tweight;
                    }
                }
            }
        }

        lock_guard<mutex> lock(globalMutex);
        for (int i = 0; i < ncoords; ++i) {
            h_datacube[i] += localDataCube[i];
            h_weightscube[i] += localWeightsCube[i];
        }

    return; 
}


/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const int &tNum) {
    double iTime1 = cpuSecond();
    // Read input points.
    //read_input_map_hdf5(infile);
    
    
    double iTime2 = cpuSecond();
    read_input_map(infile);
   
    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output();

    double iTmie6 = cpuSecond();
    printf("pre process time = %f, ", (iTmie6 - iTime2) * 1000);

    uint32_t pix_num = h_zyx[1] * h_zyx[2];
    searchcount = RALLOC(uint32_t, h_GMaps.data_shape);
    computecount = RALLOC(uint32_t, pix_num);
    memset(searchcount, 0, sizeof(uint32_t) * h_GMaps.data_shape);
    memset(computecount, 0, sizeof(uint32_t) * pix_num);
    printf("h_GMaps.data_shape %d, ", h_GMaps.data_shape);
    //    
    // Block Indirect Sort i nput points by their healpix indexes.
    // if (param == THRUST) { 
    //     init_input_with_thrust(param);
    // } else {
    //     init_input_with_cpu(param);
    // }

    printf("h_zyx[1]=%d, h_zyx[2]=%d\n", h_zyx[1], h_zyx[2]);
    // for(int i = 0; i < h_zyx[1]; i++){
    //     for(int j = 0; j < h_zyx[2]; j++){
    //         printf("%f ", h_ywcs[i*90 + j]);
    //     }
    //     printf("\n");
    // }


    // threads(bDim)
    double iTime3 = cpuSecond();
    vector<thread> threads;
    for (int i = 0; i < tNum; ++i) {
        uint32_t startIdx = i * (h_GMaps.data_shape / tNum);
        uint32_t endIdx = (i == tNum - 1) ? h_GMaps.data_shape : (i + 1) * (h_GMaps.data_shape / tNum);
        // printf("thread id %d, start %d, end %d\n", i, startIdx, endIdx);
        double *localDataCube;
        double *localWeightsCube;
        localDataCube    = RALLOC(double, pix_num);
        localWeightsCube = RALLOC(double, pix_num);
        // 创建局部副本数组
        // vector<double> localDataCube(pix_num, 0.0);
        // vector<double> localWeightsCube(pix_num, 0.0);

        // printf("datacube size is %ld\n", localWeightsCube.size());
        threads.emplace_back(hcgridThreaded,
            h_lons, h_lats, h_data, h_weights, h_xwcs, h_ywcs,
            localDataCube, localWeightsCube,
            startIdx, endIdx);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    double iTime4 = cpuSecond();
    double elapsedTime = (iTime4 - iTime3) * 1000;
    printf("kernel elapsed time = %f, ", elapsedTime);

    // Get stop time.
    // printf("kernel elapsed time=%f, ", elapsedTime);

    // Send data from GPU to CPU
    // data_d2h();

    // Write output FITS file
    printf("\n");
    write_output_map(outfile);

    // Write sorted input FITS file

//     uint32_t smin =  h_GMaps.data_shape, cmin =  pix_num;
//     uint32_t smax = 0, cmax = 0;
//     double avg_search = 0, avg_compute = 0;

//     for(int i = 0; i < pix_num; i++){
//         if(computecount[i] < cmin) cmin = computecount[i];
//         if(computecount[i] > cmax) cmax = computecount[i];
//         avg_compute += computecount[i];
//    }

//     avg_compute = avg_compute / pix_num;
    
//     printf("\nsearch count min = %d, search count max = %d, average search count = %lf\n", smin, smax, avg_search);
//     printf("\ncompute count min = %d, compute count max = %d, average ncompute count = %lf\n", cmin, cmax, avg_compute);


    // Release data
    // data_free();
    // HANDLE_ERROR( cudaEventDestroy(start) );
    // HANDLE_ERROR( cudaEventDestroy(stop) );
    // HANDLE_ERROR( cudaDeviceReset() );

    double iTime5 = cpuSecond();
    double iElaps = (iTime5 - iTime1) * 1000.;
    printf("solving_gridding time=%f\n", iElaps);
}