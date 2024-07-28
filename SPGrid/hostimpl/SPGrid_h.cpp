// --------------------------------------------------------------------
//
// title                  :HCGrid.cpp
// description            :Grid data points to map
// author                 :
//
// --------------------------------------------------------------------

#include "SPGrid_h.h"
#include "gridding.h"


int main(int argc, char **argv){
    // Get FITS files from command
    char *path = NULL, *ifile = NULL, *tfile = NULL, *ofile = NULL, *sfile = NULL, *num = NULL, *beam = NULL, *tNum = NULL, *factor = NULL;
    char pcl;
    int option_index = 0;
    static const struct option long_options[] = {
        {"helparg", no_argument, NULL, 'h'},
        {"fits_path", required_argument, NULL, 'p'},            // absolute path of FITS file
        {"input_file", required_argument, NULL, 'i'},           // name of unsorted input FITS file (it will call sort function)
        {"target_file", required_argument, NULL, 't'},          // name of target FITS file
        {"output_file", required_argument, NULL, 'o'},          // name of output FITS file
        {"fits_id", required_argument, NULL, 'n'},              // ID of FITS file
        {"beam_size", required_argument, NULL, 'b'},            // beam size of FITS file
        //{"order_arg", required_argument, NULL, 'd'},            // sort parameter
        {"thread_num", required_argument, NULL, 'a'},             // the number of thread in each block
        // {"coarsening_factor", required_argument, NULL, 'f'},    // the value of coarsening factor
        {0, 0, 0, 0}
    };

    while((pcl = getopt_long_only (argc, argv, "hp:i:t:o:s:n:b:a:", long_options, \
                    &option_index)) != EOF){
        switch(pcl){
            case 'h':
                fprintf(stderr, "useage: ./HCGrid --fits_path <absolute path> --input_file <input file> --target_file <target file> "
                "--sorted_file <sorted file> --output_file <output file> --fits_id <number> --beam_size <beam> --block_num <num>\n"
                 "example: ./HCGrid -p /home/yangwd/HCGrid-scatter/HCGrid-cpu/test_data/  -i input -t target -o output -n 5 -b 300 -a 16\n");
                return 1;
            case 'p':
                path = optarg;
                break;
            case 'i':
                ifile = optarg;
                break;
            case 't':
                tfile = optarg;
                break;
            case 'o':
                ofile = optarg;
                break;
            case 'n':
                num = optarg;
                break;
            case 'b':
                beam = optarg;
                break;
            case 'a':
                tNum = optarg;
                break;
            case '?':
                fprintf (stderr, "Unknown option `-%c'.\n", (char)optopt);
                break;
            default:
                return 1;
        }
    }

    char infile[180] = "", tarfile[180] = "", outfile[180] = "!";
    strcat(infile, path);
    strcat(infile, ifile);
    strcat(infile, num);
    strcat(infile, ".fits");
    strcat(tarfile, path);
    strcat(tarfile, tfile);
    strcat(tarfile, num);
    strcat(tarfile, ".fits");
    strcat(outfile, path);
    strcat(outfile, ofile);
    strcat(outfile, num);
    strcat(outfile, ".fits");

    printf("num: %s\n ", num);
//    printf("input file is: %s\n", infile);
//    printf("target file is: %s\n", tarfile);
//    printf("output file is: %s\n", outfile);
//    printf("sort file is: %s\n", sortfile);

    // Initialize healpix
    // _Healpix_init(1, RING);
   //  printf("\033[0m\033[1;31m%s\033[0m\n", "ERROR!");
    unsigned int max_threads = omp_get_max_threads();

    if(atoi(tNum) > max_threads){
        printf("Error: Max threads supported by the system is %d, the '-a' parameter you enter is too large\n", max_threads); 
        return  0;
    }
    // unsigned int num_cpus = thread::hardware_concurrency();

    // if (num_cpus == 0) {
    //     cout << "Unable to determine the number of CPUs." << endl;
    // } else {
    //     cout << "Number of CPUs: " << num_cpus << endl;
    // }

    // Set kernel
    uint32_t kernel_type = GAUSS1D;
    double kernelsize_fwhm = 300. / 3600.;
    if (beam) {
        double kernelsize_fwhm = atoi(beam) / 3600.;
    }
    double kernelsize_sigma = kernelsize_fwhm / sqrt(8*log(2));
    double *kernel_params;
    kernel_params = RALLOC( double, 3);
    kernel_params[0] = kernelsize_sigma;
    double sphere_radius = 5. * kernelsize_sigma;
    // double hpx_max_resolution = kernelsize_sigma / 2.;
    _prepare_grid_kernel(kernel_type, kernel_params, sphere_radius);
    
    // Gridding process
    // h_GMaps.factor = 1;
    // if (factor) {
    //     h_GMaps.factor = atoi(factor);
    // }
    // printf("h_GMaps.factor=%d,\n", h_GMaps.factor);

    // read_input_map_hdf5(infile);
    // MPI_Init(&argc, &argv);     // 初始化MPI环境
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // 获取当前进程ID
    // MPI_Comm_size(MPI_COMM_WORLD, &np);      // 获取总进程数

    
    if (tNum)
        solve_gridding(infile, tarfile, outfile, atoi(tNum));
    else
        solve_gridding(infile, tarfile, outfile, max_threads);


    return 0;
}
