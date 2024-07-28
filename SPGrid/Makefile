CXX         := gcc
INCLUDE     :=-I/usr/include/c++\
			-I/home/yangwd/HCGrid-scatter/ScatterGrid/ScatterGrid\
			-I/home/yangwd/Software/cfitsio/include\
			-I/home/yangwd/Software/wcslib/include\
			-I./
LIBRARIES   :=-L/home/yangwd/Software/wcslib/lib\
			-L/home/yangwd/Software/cfitsio/lib\

CXX_FLAGS := -std=c++11 -fopenmp
FITS_FLAGS := -lm -lcfitsio -lwcs
# CUDA_ARCH=-gencode arch=compute_70,code=sm_70
ifeq ($(USE_CUDA), 1)
	SRC_FILES := ./deviceimpl/helpers_d.o ./deviceimpl/gmap_d.o ./deviceimpl/gridding_d.o ./deviceimpl/SPGrid_d.o
	CUDA_PATH ?=/usr/local/cuda-11.2
	NVCC      :=$(CUDA_PATH)/bin/nvcc
	INCLUDE   :=$(INCLUDE)\
				-I/usr/local/cuda-11.2/include\
				-I/usr/local/cuda-11.2/samples/common/inc
	LIBRARIES :=$(LIBRARIES)-L/usr/local/cuda-11.2/lib64 -lcudart -lcufft
	CUDA_ARCH=-gencode arch=compute_70,code=sm_70     
	NVCC_FLAGS := -Wno-deprecated-gpu-targets -dc
	EXECUTABLE := SPGrid-gpu
	CXX_FLAGS := -std=c++11
else 
	SRC_FILES := ./hostimpl/helpers.o ./hostimpl/gmap.o ./hostimpl/gridding.o ./hostimpl/SPGrid_h.o
	EXECUTABLE := SPGrid-cpu
endif
# -I/usr/local/cuda-11.2/include\
# -I/usr/local/cuda-11.2/samples/common/inc


#CPU Object files
./hostimpl/helpers.o: ./hostimpl/helpers.cpp
	$(CXX) -O3 $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

./hostimpl/gmap.o: ./hostimpl/gmap.cpp
	$(CXX) -O3 $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

./hostimpl/gridding.o: ./hostimpl/gridding.cpp
	$(CXX) -O3 $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
./hostimpl/SPGrid_h.o: ./hostimpl/SPGrid_h.cpp
	$(CXX) -O3 $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

#GPU Object files
./deviceimpl/helpers_d.o: ./deviceimpl/helpers_d.cu
#		$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<
./deviceimpl/gmap_d.o: ./deviceimpl/gmap_d.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
./deviceimpl/gridding_d.o: ./deviceimpl/gridding_d.cu
#		$(NVCC) -O3 -ccbin gcc $(NVCC_FLAGS) --ptxas-options=-v $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX)  -arch=sm_70 $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $< 
./deviceimpl/SPGrid_d.o: ./deviceimpl/SPGrid_d.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<


$(EXECUTABLE): $(SRC_FILES)
ifeq ($(USE_CUDA), 1)
	@ echo ./$@ $+
	$(NVCC) -O3 -ccbin $(CXX) -Wno-deprecated-gpu-targets $(INCLUDE) -o $@ $+ $(CUDA_ARCH) $(LIBRARIES) $(FITS_FLAGS) $(CXX_FLAGS)
else
	@ echo ./$@ $+
	$(CXX) -O3 $(INCLUDE) -o $@ $+ $(LIBRARIES) $(FITS_FLAGS) $(CXX_FLAGS) -lstdc++ 
endif

all: $(EXECUTABLE)
clean:
	rm -rf ./hostimpl/*.o ./deviceimpl/*.o $(EXECUTABLE)