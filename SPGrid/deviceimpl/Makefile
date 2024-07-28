CC          := clang
CXX         := gcc
AR          := ar -rc
CUDA_PATH   ?=/usr/local/cuda-11.2
NVCC        :=$(CUDA_PATH)/bin/nvcc
INCLUDE     :=-I/usr/include/c++\
			-I/home/yangwd/HCGrid-scatter//HCGrid-gpu/HCGrid\
			-I/home/yangwd/Software/cfitsio/include\
			-I/home/yangwd/Software/wcslib/include\
			-I./\
			-I/usr/local/cuda-11.2/include\
			-I/usr/local/cuda-11.2/samples/common/inc
LIBRARIES   :=-L/usr/local/cuda-11.2/lib64 -lcudart -lcufft\
			-L/home/yangwd/Software/wcslib/lib\
			-L/home/yangwd/Software/cfitsio/lib\
# -arch=sm_50 \ 
# 		  -gencode=arch=compute_50,code=sm_50 \ 
# 		  -gencode=arch=compute_52,code=sm_52 \ 
# 		  -gencode=arch=compute_60,code=sm_60 \ 
# 		  -gencode=arch=compute_61,code=sm_61 \ 
# 		  -gencode=arch=compute_70,code=sm_70 \ 
# 		  -gencode=arch=compute_75,code=sm_75 \
# 		  -gencode=arch=compute_75,code=compute_75 
# -gencode arch=compute_35,code=sm_35 \
#  		  -gencode arch=compute_50,code=sm_50 \
# 		  -gencode arch=compute_60,code=sm_60 \

CUDA_ARCH=-gencode arch=compute_70,code=sm_70     

NVCC_FLAGS := -Wno-deprecated-gpu-targets -dc
CXX_FLAGS := -std=c++11
FITS_FLAGS := -lm -lcfitsio -lwcs

helpers_d.o: helpers_d.cu
#	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<

gmap_d.o: gmap_d.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

gridding_d.o: gridding_d.cu
#	$(NVCC) -O3 -ccbin gcc $(NVCC_FLAGS) --ptxas-options=-v $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX)  -arch=sm_70 $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $< 

ScatterGrid_d.o: ScatterGrid_d.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

ScatterGrid-gpu: helpers_d.o gmap_d.o gridding_d.o ScatterGrid_d.o
	@ echo ./$@ $+
	$(NVCC) -O3 -ccbin $(CXX) -Wno-deprecated-gpu-targets $(INCLUDE) -o $@ $+ $(CUDA_ARCH) $(LIBRARIES) $(FITS_FLAGS) $(CXX_FLAGS)
	
clean:
	rm -rf *.o ScatterGrid-gpu
